import os,shutil
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from document_store import ingest_documents, search_documents,load_vectorstore


app = FastAPI()
UPLOAD_DIR="./uploads"
os.makedirs(UPLOAD_DIR,exist_ok=True)

# Load Vector Store on startup
retriever = load_vectorstore()



@app.get("/")
def read_root():
    return {"Hello":"World"}


