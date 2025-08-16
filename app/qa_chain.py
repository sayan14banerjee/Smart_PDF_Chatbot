import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ✅ Import helper from utils.py
from app.utils import load_faiss_index


def load_vector_store():
    """Load FAISS vector store via utils.py"""
    print("Loading FAISS index...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ Use centralized helper
    return load_faiss_index("data/faiss_index", embeddings)


def initialize_instruction_model():
    """Load a local instruction-tuned model"""
    model_name = "google/flan-t5-base"  # small and instruction tuned

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        temperature=0,
        do_sample=False,
        device=0 if torch.cuda.is_available() else -1
    )

    return HuggingFacePipeline(pipeline=pipe)


def create_qa_chain(vector_store, llm):
    """Create Retrieval QA chain with custom prompt"""
    prompt_template = """
You are a helpful assistant. Use the provided context to answer the question concisely.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


def get_qa_chain():
    vector_store = load_vector_store()
    llm = initialize_instruction_model()
    qa_chain = create_qa_chain(vector_store, llm)

    print("\n✅ Smart PDF Chatbot is ready!")
    print("Ask questions about your document. Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break

        result = qa_chain.invoke({"query": question})

        print(f"\nAnswer: {result['result']}\n")

        if result.get("source_documents"):
            print("Sources:")
            for i, doc in enumerate(result["source_documents"], 1):
                snippet = doc.page_content.strip().replace("\n", " ")
                print(f"{i}. {snippet[:150]}...")
        print("-" * 50)


if __name__ == "__main__":
    get_qa_chain()
