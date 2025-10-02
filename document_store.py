import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document
from typing import List


def get_embeddings(device="cpu"):
    return HuggingFaceEmbeddings(model_name="thenlper/gte-small", model_kwargs = {"device":device})
def load_files_to_docs(paths: List[str]):
    """Load files into Langchain document object (takes file urls right now)"""

    docs = []

    for p in paths:
        ext = p.split(".")[-1].lower()

        if(ext == "pdf"):
            loader = PyPDFLoader(p)
            loaded = loader.load()


        elif (ext in ("txt","md")):
            loader = TextLoader(p)
            loaded = loader.load(p,encoding="utf-8")


        else:
            try:
                loader = TextLoader(p)
                loaded = loader.load(p,encodings = "utf-8")

            except Exception:
                print(f"Warning: File type not supported : {p}")
                loaded = []

        docs.extend(loaded)


    return docs



def ingest_documents(paths: List[str], persist_dir: str = "vector_store", device: str = "cpu"):
    """Main Orchestrator function that handles the document ingestion functionality. It uses load_files_to_docs() function to convert files to
        Langchain docs, then breaks them to chunksk of suitable size, then stores them in FAISS vector index, then returns that index.
    """

    os.makedirs(persist_dir,exist_ok=True)
    docs = load_files_to_docs(paths)

    if (not docs):
        raise ValueError("No documents loaded from the given paths.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings(device)
    index = FAISS.from_documents(chunks,embeddings)
    index.save_local(persist_dir)
    return index


def load_vectorstore(persist_dir: str = "vector_store", device: str = "cpu"):

    """Loads the existing FAISS index, then returns a retriever so that we can use the existing vector store as a retriever to retrieve
    embeddings that match user query.
    """

    embeddings = get_embeddings(device)

    if not os.path.exists(persist_dir):
        return None

    index = FAISS.load_local(persist_dir,embeddings, allow_dangerous_deserialization=True)
    retriever = index.as_retriever(search_kwargs = {"k":4})
    return retriever



def search_documents(retriever,query: str, k: int = 4):
    """Searches the retriever and gets top 'k' (4) documents that match user's query."""

    if retriever is None:
        return []

    retriever.search_kwargs["k"] = k
    return retriever.get_relevant_documents(query)


