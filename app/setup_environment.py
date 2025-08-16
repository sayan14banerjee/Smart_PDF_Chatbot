import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    packages = [
        "langchain",
        "langchain-community", 
        "langchain-huggingface",
        "huggingface_hub",
        "sentence-transformers",
        "faiss-cpu",
        "pypdf",
        "transformers",
        "torch"
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")

def check_environment():
    """Check environment setup"""
    print("\nEnvironment Check:")
    print("-" * 30)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check for HuggingFace token
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        print("✓ HuggingFace API token found")
    else:
        print("⚠ HuggingFace API token not found")
        print("  You can get one from: https://huggingface.co/settings/tokens")
        print("  Set it as: export HUGGINGFACEHUB_API_TOKEN=your_token_here")
    
    # Check for PDF file
    pdf_path = "data/document.pdf"
    if os.path.exists(pdf_path):
        print(f"✓ PDF file found at {pdf_path}")
    else:
        print(f"⚠ PDF file not found at {pdf_path}")
        print("  Please place your PDF file in the data/ directory")

if __name__ == "__main__":
    install_requirements()
    check_environment()
