from langchain.text_splitter import RecursiveCharacterTextSplitter #type: ignore
import os

def split_text(text, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Splits a text file into smaller chunks and saves them.
    
    Args:
        input_path (str): Path to the input text file.
        output_path (str): Path to save the split chunks.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks to maintain context.
    """
    
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    
    return chunks


if __name__ == "__main__":
    input_file = "data/pdf_text.txt"
    output_file = "data/chunks.txt"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read original text
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = split_text(text) # type: ignore
    # Save chunks
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n")
            f.write(chunk + "\n\n")
    
    print(f"✅ Text split into {len(chunks)} chunks and saved to {output_file}")
