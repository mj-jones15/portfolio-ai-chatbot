"""
chatbot.py — FastAPI backend for the portfolio RAG chatbot.

Startup is fast: loads the pre-embedded ChromaDB from disk (built by ingest.py).
No documents are read or embedded at runtime — only the query gets embedded.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from pathlib import PATH, Path
import psutil
import os

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Text, DateTime, func

from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.llms.together import TogetherLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"
MODEL_NAME  = "all-mpnet-base-v2"

# ── Query-logging DB (Postgres on Railway, SQLite locally) ────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///queries_local.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

_engine_kwargs = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

db_engine  = create_engine(DATABASE_URL, **_engine_kwargs)
db_metadata = MetaData()

query_logs = Table(
    "query_logs", db_metadata,
    Column("id",        Integer, primary_key=True),
    Column("query",     Text,    nullable=False),
    Column("response",  Text,    nullable=False),
    Column("timestamp", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

with db_engine.begin() as _conn:
    db_metadata.create_all(_conn)

# ── LlamaIndex settings ───────────────────────────────────────────────────────

Settings.embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME)

# Together AI Model
Settings.llm = TogetherLLM(
    model="openai/gpt-oss-20b",
    api_key=os.environ["TOGETHER_API_KEY"],
    temperature=0.34,
)

# ── Load pre-embedded vector store from disk ──────────────────────────────────

print("[INFO] Loading ChromaDB from disk...")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
# Debugging
print("[INFO] Available collections:")
print(chroma_client.list_collections())
# End debugging
collection    = chroma_client.get_collection("filesAboutMyself")
vector_store  = ChromaVectorStore(chroma_collection=collection)
storage_ctx   = StorageContext.from_defaults(vector_store=vector_store)

# from_vector_store() reads the existing index — does NOT re-embed anything
index = VectorStoreIndex.from_vector_store(vector_store)

process = psutil.Process(os.getpid())
mem_mb  = process.memory_info().rss / 1024 / 1024
print(f"[INFO] ChromaDB loaded. Memory: {mem_mb:.2f} MB")

# ── Custom prompt ─────────────────────────────────────────────────────────────

PROMPT = (
    "You are a personal AI assistant representing Matthew Jones, a Computer Science junior "
    "at the University of Kentucky with a strong focus on AI research and AI policy. "
    "Your purpose is to answer questions about Matthew's skills, experience, career aspirations, "
    "and projects based on the provided context below. "
    "Always answer in the first person, as if you are Matthew — use 'I have experience in...' "
    "rather than 'Matthew has experience in...'. "
    "Be friendly, professional, and confident.\n\n"
    "A little about me:\n"
    "- Junior at the University of Kentucky studying Computer Science, "
      "minors in Mathematics and Vocal Performance\n"
    "- Pursuing an AI certificate; career goal is a PhD in CS specializing in AI and public policy, "
      "with aspirations to become a legislative advisor on AI\n"
    "- AI Researcher at the UK Center for Computational Sciences, supporting the NSF ACCESS ecosystem\n"
    "- Studied abroad in Bilbao, Spain in Spring 2025\n"
    "- Fluent in Spanish\n"
    "---------------------\n"
    "Context: {context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)

qa_template  = PromptTemplate(PROMPT)
query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=5)

# ── Helper ────────────────────────────────────────────────────────────────────

def get_rag_response(user_input: str) -> str:
    response = query_engine.query(user_input)
    return response.response if hasattr(response, "response") else str(response)

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://portfolio-frontend-enf4.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI"}

@app.post("/chat")
async def chat(query: UserQuery):
    print(f"[QUERY] {query.query}")
    response = get_rag_response(query.query)

    mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"[DEBUG] Memory: {mem_mb:.2f} MB")

    try:
        with db_engine.begin() as conn:
            conn.execute(query_logs.insert().values(
                query=query.query,
                response=str(response),
            ))
    except Exception as e:
        print(f"[WARN] Failed to log query: {e}")

    return {"response": str(response)}