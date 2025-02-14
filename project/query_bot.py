# query_bot.py
import faiss
import numpy as np
import pickle
import requests
from sentence_transformers import SentenceTransformer

# 1. Load FAISS index and chunks
index = faiss.read_index("faiss_index.index")
with open("chunks.pkl", "rb") as f:
    all_text_chunks = pickle.load(f)

# 2. Load the same embedding model used for indexing
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Which model do you want Ollama to use?
# Make sure you have downloaded it in Ollama (e.g. "ollama run llama3.2" once).
OLLAMA_MODEL = "llama2:7b-chat"  # Example name — adapt to your actual model tag

def retrieve_chunks(question, top_k=3):
    """
    Given a user question, return the top_k relevant chunks from your index.
    """
    q_embedding = embedder.encode(question).astype('float32').reshape(1, -1)
    distances, indices = index.search(q_embedding, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        filename, chunk_text = all_text_chunks[idx]
        results.append((dist, filename, chunk_text))
    return results

def call_ollama(prompt, model=OLLAMA_MODEL):
    """
    Calls the local Ollama server at http://localhost:11434/generate
    with a given prompt and model. Returns the text output.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        # Optionally, add more config like: "temperature": 0.2, "max_tokens": 300
        "temperature": 0.5,
        "max_tokens": 256
    }
    response = requests.post(url, json=payload, stream=True)
    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.text}")
    
    # Ollama streams the response line by line
    generated_text = []
    for line in response.iter_lines():
        if line:
            # Each line is a JSON with {"done": bool, "response": "..."} or similar
            data = line.decode('utf-8')
            try:
                # Example: {"response":"some text", "done":false}
                json_data = eval(data) if data.startswith('{') else {}
                # We parse the "response" field
                if "response" in json_data:
                    generated_text.append(json_data["response"])
            except:
                continue
    
    return "".join(generated_text)

def main():
    user_question = input("Ask a question about Status: ")
    top_chunks = retrieve_chunks(user_question, top_k=3)

    # Build the context text from your retrieved chunks
    context_text = ""
    for dist, fn, chunk in top_chunks:
        context_text += f"From {fn}:\n{chunk}\n\n"

    # Define your prompt (or system_prompt)
    prompt = f"""
You are a helpful assistant with knowledge only from the text below. 
If the answer isn't in the text, say "I don't know based on the available documentation."

Context:
{context_text}

User question: {user_question}

Answer:
"""

    # Now it’s safe to print it!
    print("DEBUG: Final prompt to Ollama:\n", prompt)

    # Finally, call your function to query Ollama
    answer = call_ollama(prompt, model=OLLAMA_MODEL)
    print("-----")
    print("Answer:", answer.strip())

if __name__ == "__main__":
    main()
