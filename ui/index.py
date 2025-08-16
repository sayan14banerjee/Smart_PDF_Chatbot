import streamlit as st
import os
import tempfile
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Import your backend modules
import os
import sys

# Add parent directory so Python can find "app"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now imports will work
from app.pdf_handler import extract_text_from_pdf
from app.text_splitter import split_text
from app.utils import save_faiss_index, load_faiss_index
from app.qa_chain import create_qa_chain, initialize_instruction_model
from app.embedder import create_faiss_index

# ------------------------------
# Streamlit Configuration
# ------------------------------

st.set_page_config(
    page_title="📄 Smart PDF Chatbot", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Helper Functions
# ------------------------------

def create_faiss_from_chunks(chunks, index_path="data/faiss_index"):
    """Create FAISS index directly from text chunks"""
    try:
        # Convert chunks to Document objects
        documents = [Document(page_content=chunk) for chunk in chunks if chunk.strip()]
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create FAISS index
        from langchain_community.vectorstores import FAISS
        vector_store = FAISS.from_documents(documents, embeddings)
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save the index
        save_faiss_index(vector_store, index_path)
        
        return vector_store
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False

# ------------------------------
# Main Streamlit App
# ------------------------------

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("📄 Smart PDF Chatbot")
    st.markdown("Upload a PDF document and ask questions about its content using AI!")
    
    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # File uploader
        uploaded_pdf = st.file_uploader(
            "Choose a PDF file", 
            type=["pdf"],
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_pdf is not None:
            st.success(f"✅ File uploaded: {uploaded_pdf.name}")
            
            # Process PDF button
            if st.button("🔄 Process PDF", type="primary"):
                process_pdf(uploaded_pdf)
        
        # Display processing status
        if st.session_state.pdf_processed:
            st.success("✅ PDF processed and ready for questions!")
        
        # Reset button
        if st.button("🗑️ Reset Chat"):
            reset_session()
    
    # Main content area
    if st.session_state.pdf_processed and st.session_state.qa_chain:
        # Chat interface
        st.header("💬 Ask Questions")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer)
        
        # Question input
        user_question = st.chat_input("Type your question about the PDF...")
        
        if user_question:
            # Add user question to chat
            with st.chat_message("user"):
                st.write(user_question)
            
            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        result = st.session_state.qa_chain.invoke({"query": user_question})
                        answer = result['result']
                        
                        st.write(answer)
                        
                        # Show sources if available
                        if result.get("source_documents"):
                            with st.expander("📚 View Sources"):
                                for i, doc in enumerate(result["source_documents"], 1):
                                    snippet = doc.page_content.strip().replace("\n", " ")
                                    st.write(f"**Source {i}:** {snippet[:200]}...")
                        
                        # Add to chat history
                        st.session_state.chat_history.append((user_question, answer))
                        
                    except Exception as e:
                        error_msg = f"❌ Error generating answer: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append((user_question, error_msg))
    
    else:
        # Welcome message
        st.info("👆 Please upload a PDF file in the sidebar to get started!")
        
        # Instructions
        st.markdown("""
        ### How to use:
        1. **Upload PDF**: Click on the file uploader in the sidebar
        2. **Process**: Click the "Process PDF" button to analyze the document
        3. **Ask Questions**: Once processed, you can ask questions about the content
        4. **View Sources**: Expand the sources section to see relevant document excerpts
        """)

def process_pdf(uploaded_pdf):
    """Process the uploaded PDF file"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.getvalue())
            tmp_file_path = tmp_file.name
        
        # Step 1: Extract text from PDF
        with st.spinner("📑 Extracting text from PDF..."):
            text = extract_text_from_pdf(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        if not text or not text.strip():
            st.error("❌ Could not extract text from this PDF. Please try a different file.")
            return
        
        st.success("✅ Text extracted successfully!")
        
        # Step 2: Split text into chunks
        with st.spinner("✂️ Splitting text into chunks..."):
            chunks = split_text(text, chunk_size=500, chunk_overlap=50)
        
        st.success(f"✅ Created {len(chunks)} text chunks!")
        
        # Step 3: Create FAISS index
        with st.spinner("🔍 Creating vector embeddings..."):
            vector_store = create_faiss_from_chunks(chunks)
        
        if vector_store is None:
            st.error("❌ Failed to create vector embeddings.")
            return
        
        st.success("✅ Vector embeddings created!")
        
        # Step 4: Initialize QA chain
        with st.spinner("🤖 Initializing AI model..."):
            llm = initialize_instruction_model()
            qa_chain = create_qa_chain(vector_store, llm)
        
        # Store in session state
        st.session_state.vector_store = vector_store
        st.session_state.qa_chain = qa_chain
        st.session_state.pdf_processed = True
        st.session_state.chat_history = []  # Reset chat history
        
        st.success("🎉 PDF processed successfully! You can now ask questions.")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        st.session_state.pdf_processed = False

def reset_session():
    """Reset the session state"""
    st.session_state.vector_store = None
    st.session_state.qa_chain = None
    st.session_state.chat_history = []
    st.session_state.pdf_processed = False
    st.success("🔄 Session reset successfully!")
    st.rerun()

# ------------------------------
# Run the app
# ------------------------------

if __name__ == "__main__":
    main()
