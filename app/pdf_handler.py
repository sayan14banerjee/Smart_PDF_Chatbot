import fitz  # PyMuPDF
import os

# Path to data directory (one level up from app/)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file and save it into data/ folder."""
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        print(f"✅ Opened PDF: {pdf_path}")
        
        all_text = ""
        for page_num, page in enumerate(doc, start=1): #type: ignore
            # Extract text from each page
            text = page.get_text()
            all_text += f"\n--- Page {page_num} ---\n{text}"
        
       

        
        return all_text
    except Exception as e:
        print(f"❌ Error extracting text: {e}")

if __name__ == "__main__":
    # Change this to your test PDF path
    test_pdf = os.path.join(os.path.dirname(__file__), "..", "internship.pdf")
    
    if os.path.exists(test_pdf):
        all_text = extract_text_from_pdf(test_pdf)
    
    else:
        print(f"❌ Test PDF not found at {test_pdf}")

    output_file="pdf_text.txt"
    if not all_text:
        print("❌ No text extracted from PDF.")
        exit(1)
     # Ensure data folder exists
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, output_file)
    # Save extracted text
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(all_text) # type: ignore
    
    print(f"✅ Text extracted and saved to: {output_file}")
