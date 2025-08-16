import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face API token not found in .env file.")

# Initialize Hugging Face LLM
def initialize_llm():
    print("⚡ Initializing Hugging Face LLM...")
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",  # Safe free model
        temperature=0.1,
        max_new_tokens=512
    ) # type: ignore
    print("✅ Hugging Face LLM ready!")
    return llm

# Load FAISS index
def load_vectorstore(index_path="data/faiss_index"):
    print("✅ Loading FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Create QA chain
def create_qa_chain(llm, vector_store):
    print("🔍 Creating QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    print("✅ QA Chain ready!")
    return qa_chain

def main():
    vector_store = load_vectorstore()
    llm = initialize_llm()
    qa_chain = create_qa_chain(llm, vector_store)

    print("\n🤖 Smart PDF Chatbot is ready!")
    print("Ask questions about your document. Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ")
        if question.lower() == "exit":
            print("Goodbye! 👋")
            break

        print("💭 Thinking...")
        try:
            result = qa_chain.invoke({"query": question})
            print("\nAnswer:", result["result"])
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
