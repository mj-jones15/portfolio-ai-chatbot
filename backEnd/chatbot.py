from fastapi import FastAPI, Request
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import chromadb
import psutil
import os
from chromadb.utils import embedding_functions
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("chatbot:app", host="0.0.0.0", port=port)

model_name = "all-mpnet-base-v2"


#Custom embedding function with the loaded model
class chosenEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, texts):
        return model.encode(texts, convert_to_tensor=True).tolist() # type: ignore


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

def load_vector_index():
    global index, query_engine

    model = SentenceTransformer(model_name)

    if query_engine is not None:
        return

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

    #Llama3 model
    modelName = "llama3"
    Settings.llm = Ollama(modelName, request_timeout=360.0)

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

    #Query engine
    query_engine = index.as_query_engine()


#Set up port
app = FastAPI()

origins = [
    "https://matthewjones-ukcs.onrender.com",  # adjust this to your frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class UserQuery(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    load_vector_index()

@app.post("/chat")
async def chat(query: UserQuery):
    print(f"Recieved query: {query.query}")
    response = get_rag_response(query.query)

    # Track memory usage
    process = psutil.Process(os.getpid())
    memory_used_mb = process.memory_info().rss / 1024 / 1024
    print(f"[DEBUG] Memory usage: {memory_used_mb:.2f} MB")


    return {"response": str(response)}
