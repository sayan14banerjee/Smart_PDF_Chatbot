# app/utils.py
from langchain_community.vectorstores import FAISS

def save_faiss_index(vector_store, index_path: str):
    """
    Saves a FAISS vector store to disk.
    
    Args:
        vector_store: FAISS vectorstore object
        index_path (str): Directory where FAISS index will be stored
    """
    vector_store.save_local(index_path)
    print(f"✅ FAISS index saved at {index_path}")


def load_faiss_index(index_path: str, embeddings=None):
    """
    Loads a FAISS vector store from disk.
    
    Args:
        index_path (str): Path where FAISS index is stored
        embeddings: Embedding function (required if using new embeddings)
    Returns:
        FAISS vectorstore object
    """
    try:
        vector_store = FAISS.load_local(
            index_path,
            embeddings=embeddings, # type: ignore
            allow_dangerous_deserialization=True
        )
        print(f"✅ FAISS index loaded from {index_path}")
        return vector_store
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load FAISS index: {e}")
