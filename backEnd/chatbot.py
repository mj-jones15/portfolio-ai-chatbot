from fastapi import FastAPI, Request
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import chromadb
import psutil
import os
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Text, DateTime, func
from chromadb.utils import embedding_functions
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings, PromptTemplate
from llama_index.llms.together import TogetherLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer


model_name = "all-mpnet-base-v2"

model = SentenceTransformer(model_name)

# --- Query logging DB (Postgres via SQLAlchemy) ---
# Use DATABASE_URL from env (Railway). Fallback to local SQLite when running locally.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///queries_local.db")

# Railway sometimes exposes URLs like postgres://... ; SQLAlchemy expects postgresql+psycopg2://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

_engine_kwargs = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

db_engine = create_engine(DATABASE_URL, **_engine_kwargs)
db_metadata = MetaData()

query_logs = Table(
    "query_logs",
    db_metadata,
    Column("id", Integer, primary_key=True),
    Column("query", Text, nullable=False),
    Column("response", Text, nullable=False),
    Column("timestamp", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

# Create the table if it doesn't exist
with db_engine.begin() as _conn:
    db_metadata.create_all(_conn)


#Custom embedding function with the loaded model
class chosenEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, texts):
        return model.encode(texts, convert_to_tensor=True).tolist() 


# Some LlamaIndex versions return an Response object instead of a plain string
#Normal option:
#def get_rag_response(user_input: str):
#   return query_engine.query(user_input).response
def get_rag_response(user_input: str):
    response = query_engine.query(user_input)
    return response.response if hasattr(response, "response") else str(response)


# Define global variables
index = None
query_engine = None

#ChromaDB Client
embedding_function = chosenEmbeddingFunction()

#Create instance of client class 
client = chromadb.Client()

#Create new collection by calling client method
name = "filesAboutMyself"
collection = client.get_or_create_collection(name=name, embedding_function=embedding_function)

#Safety: clear all old documents from collection
collection.delete(where={"title": {"$ne": ""}})

#In case an exact deletion is required
# collection.delete(where={"title": "resume.txt"})
# path to the folder with files
dataDir = "./ragData"

#prepare lists to add to ChromaDB
documents = []
metadatas = []
ids = []

#Loop through documents in dataDir folder. Reads first line as the metadatas
for filename in os.listdir(dataDir):
    if filename.endswith(".txt"):
        filepath = os.path.join(dataDir, filename)
        with open(filepath, "r", encoding = "utf-8") as f:
            lines = f.readlines()
            if len(lines) < 2:
                print(f"Skipping {filename} — not enough lines")
                continue

            #first line is metadata
            keywords = [keyword.strip() for keyword in lines[0].split(",")]
            content = "".join(lines[1:]).strip()

            documents.append(content)
            metadatas.append({
                "title": filename,
                "keywords": ", ".join(keywords)  # convert list to a single comma-separated string
            })
            ids.append(filename)  # FIXED: assign ID per file


#Function adds items to collection, documents are added with metadata
collection.add(
    documents = documents,
    metadatas = metadatas,
    ids = ids
)
process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / 1024 / 1024
print(f"[DEBUG] Memory used after indexing: {mem_mb:.2f} MB")

#Convert ChromaDB to VectorStore
vector_store = ChromaVectorStore(chroma_collection=collection)

#Wrap vector store in storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)

# Mistral via Together API
# Mistral via Together API
Settings.llm = TogetherLLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=os.environ["TOGETHER_API_KEY"],
)

# Create Document objects from the loaded documents and metadata
doc_objects = [
    Document(text=doc_text, metadata=metadata)
    for doc_text, metadata in zip(documents, metadatas)
]

# Build the vector index
index = VectorStoreIndex.from_documents(
    documents=doc_objects,
    storage_context=storage_context,
)


# Your custom system prompt
new_prompt_str = (
    "You are a personal AI assistant representing Matthew Jones, a skilled Junior at the University of Kentucky. "
    "Your purpose is to answer questions about Matthew's skills, experience, career aspirations, and projects based on the provided context below. "
    "Always answer in the first person, as if you are Matthew. For example, use 'I have experience in...' instead of 'Matthew has experience in...'. "
    "Be friendly, professional, and confident in your abilities.\n\n"
    "---------------------\n"
    "Context: {context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_template = PromptTemplate(new_prompt_str)

# Create the query engine with the custom prompt
query_engine = index.as_query_engine(text_qa_template=qa_template)


#Set up port
app = FastAPI()

origins = [
    "https://portfolio-frontend-enf4.onrender.com",  # adjust this to your frontend URL
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class UserQuery(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Hello from Fast API"}

@app.post("/chat")
async def chat(query: UserQuery):
    print(f"Recieved query: {query.query}")
    response = get_rag_response(query.query)

    # Track memory usage
    process = psutil.Process(os.getpid())
    memory_used_mb = process.memory_info().rss / 1024 / 1024
    print(f"[DEBUG] Memory usage: {memory_used_mb:.2f} MB")

        # Persist query + response to the DB
    try:
        with db_engine.begin() as conn:
            conn.execute(
                query_logs.insert().values(
                    query=query.query,
                    response=str(response),
                )
            )
    except Exception as e:
        print(f"[WARN] Failed to log query: {e}")
    return {"response": str(response)}
