# app/embedder.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from app.utils import save_faiss_index

load_dotenv()

def create_faiss_index(input_path: str, index_path: str, use_openai: bool = False):
    """
    Creates a FAISS vector index from text chunks.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Chunks file not found: {input_path}")

    # Read chunks
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into documents
    chunks = [chunk.strip() for chunk in content.split("--- Chunk") if chunk.strip()]
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Choose embeddings
    if use_openai:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in .env file")
        print("🔹 Using OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key) # type: ignore
    else:
        print("🔹 Using HuggingFace embeddings (offline)...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS index
    vector_store = FAISS.from_documents(documents, embeddings)

    # Save FAISS index using utils
    save_faiss_index(vector_store, index_path)


if __name__ == "__main__":
    # Example run
    create_faiss_index("data/chunks.txt", "data/faiss_index", use_openai=False)
