import fitz # PyMuPDF

def extract_text_from_pdf(pdf_path):

    try:
        doc = fitz.open(pdf_path)
        all_text = ""
        for page_num , page in enumerate(doc, start=1): # type: ignore
            text = page.get_text()
            all_text += f"Page {page_num}:\n{text}\n"
        
        # Save to file
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(all_text)

        print("✅ PDF text extracted and saved to output.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
# Example usage
if __name__ == "__main__":
    extract_text_from_pdf("sample.pdf")  # Replace with your PDF file
