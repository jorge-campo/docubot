# build_index.py
import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

DOCS_PROCESSED_PATH = "docs/processed"

def load_text_files(path):
    """Load all .txt files from the given path and return a list of (filename, text)."""
    data = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            filepath = os.path.join(path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            data.append((filename, text))
    return data

def chunk_text(text, chunk_size=500):
    """
    Split text into chunks of ~300 words each.
    You can adjust chunk_size based on experimentation.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    count = 0
    
    for w in words:
        current_chunk.append(w)
        count += 1
        if count >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            count = 0
    # Add the remaining words as the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def main():
    # 1. Load docs
    docs = load_text_files(DOCS_PROCESSED_PATH)
    
    # 2. Initialize embedding model
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Lists to store chunk data
    all_text_chunks = []
    all_embeddings = []
    
    # 3. For each document, chunk & embed
    for (filename, content) in docs:
        # skip empty files
        if not content.strip():
            continue
        
        chunks = chunk_text(content, chunk_size=800)
        
        for chunk in chunks:
            # Get embedding for chunk
            embedding = embedder.encode(chunk)
            all_text_chunks.append((filename, chunk))
            all_embeddings.append(embedding)
    
    # Convert embeddings to a NumPy array
    embeddings_np = np.array(all_embeddings, dtype='float32')
    dimension = embeddings_np.shape[1]  # e.g., 384 for MiniLM
    
    # 4. Create & populate FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    # 5. Save the index and the chunk data
    faiss.write_index(index, "faiss_index.index")
    
    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_text_chunks, f)

if __name__ == "__main__":
    main()
