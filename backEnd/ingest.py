"""
ingest.py — Run this LOCALLY before every deployment.

Usage:
    python ingest.py

What it does:
  - Reads all .txt files from ./ragData (skipping files in SKIP_FILES)
  - Chunks each document using LlamaIndex's SentenceSplitter
  - Embeds chunks via HuggingFace (all-mpnet-base-v2)
  - Persists the ChromaDB vector store to ./chroma_db/

After running, commit the chroma_db/ folder to git.
Railway will bake it into the deployment — chatbot.py just loads it at startup.
"""

import os
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# ── Config ────────────────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(_HERE, "chroma_db")
DATA_DIR    = os.path.join(_HERE, "ragData")
MODEL_NAME  = "all-mpnet-base-v2"

# Files to exclude from the vector store.
# Add any example essays or noisy docs here.
SKIP_FILES = {
    # e.g. "example_essay.txt",
    #      "sample_writing.txt",
}

# Chunking config — 512 tokens with 64-token overlap is a solid default.
# Increase chunk_size if your docs are dense reference material,
# decrease if you want more precise retrieval on short factual questions.
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

# ── Embedding model (same one used at query time in chatbot.py) ───────────────

Settings.embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME)
Settings.llm = None  # No LLM needed during ingestion

# ── ChromaDB persistent client ────────────────────────────────────────────────

client = chromadb.PersistentClient(path=CHROMA_PATH)

# Always start fresh so re-running ingest doesn't create duplicate chunks
try:
    client.delete_collection("filesAboutMyself")
    print("[INFO] Deleted existing collection — starting fresh.")
except Exception:
    pass

collection   = client.get_or_create_collection("filesAboutMyself")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_ctx  = StorageContext.from_defaults(vector_store=vector_store)

# ── Load and parse documents ──────────────────────────────────────────────────

docs = []
skipped = []

for i, filename in enumerate(sorted(os.listdir(DATA_DIR)), start=1):
    if not filename.endswith(".txt"):
        continue
    if filename in SKIP_FILES:
        skipped.append(filename)
        continue

    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 2:
        print(f"[SKIP] {filename} — not enough content (needs keyword line + body)")
        continue

    # First line = comma-separated keywords (your existing convention)
    keywords = lines[0].strip()
    content  = "".join(lines[1:]).strip()

    docs.append(Document(
        text=content,
        metadata={"title": f"Document {i}", "keywords": keywords}
    ))
    print(f"[LOAD] {filename}")

if skipped:
    print(f"\n[SKIP] Excluded files: {', '.join(skipped)}")

print(f"\n[INFO] Loaded {len(docs)} document(s). Chunking and embedding...\n")

# ── Chunk + embed + persist ───────────────────────────────────────────────────

splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage_ctx,
    transformations=[splitter],
    show_progress=True,
)

storage_ctx.persist()
print(f"\n[DONE] Vector store saved to '{CHROMA_PATH}/'")
print("       Commit the chroma_db/ folder to git before deploying.")