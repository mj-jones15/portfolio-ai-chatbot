from fastapi import FastAPI, Request
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import chromadb
import os
from chromadb.utils import embedding_functions
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer




model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)

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

app = FastAPI()

origins = [
    "http://127.0.0.1:5500",  # adjust this to your frontend URL
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



@app.post("/chat")
async def chat(query: UserQuery):
    print(f"Recieved query: {query.query}")
    response = get_rag_response(query.query)
    return {"response": str(response)}