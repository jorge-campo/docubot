# query_bot.py
import faiss
import numpy as np
import pickle
import requests
import json  # Added missing import
from sentence_transformers import SentenceTransformer

# 1. Load FAISS index and chunks
index = faiss.read_index("faiss_index.index")
with open("chunks.pkl", "rb") as f:
    all_text_chunks = pickle.load(f)

# 2. Load the same embedding model used for indexing
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Which model do you want Ollama to use?
OLLAMA_MODEL = "phi4:latest"  # Example name — adapt to your actual model tag

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
    Calls the local Ollama server at http://localhost:11434/api/generate
    with a given prompt and model. Returns the text output.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.3,
        "max_tokens": 64
    }
    response = requests.post(url, json=payload, stream=True)
    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.text}")
    
    # Ollama streams the response line by line
    generated_text = []
    for line in response.iter_lines():
        if line:
            data = line.decode('utf-8')
            try:
                json_data = json.loads(data)
                if "response" in json_data:
                    generated_text.append(json_data["response"])
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e} | Raw data: {data}")
                continue
    
    return "".join(generated_text)

def main():
    user_question = input("Ask a question about Status: ")
    
    # Retrieve the top chunks
    top_chunks = retrieve_chunks(user_question, top_k=3)
    
    # Debug: Print each retrieved chunk
    print("DEBUG: Retrieved the following chunks for your question:")
    for dist, fn, chunk in top_chunks:
        print(f"File: {fn} | Distance: {dist}")
        print("Sample chunk text:\n", chunk[:300], "...")
        print("-" * 60)


    # Before building context_text, filter chunks by similarity threshold
    # MIN_SIMILARITY = 0.7  # Adjust based on your embedding space
    # filtered_chunks = [c for c in top_chunks if c[0] <= MIN_SIMILARITY]

    # If no chunks meet threshold, force "I don't know" response
    # if not filtered_chunks:
    #    return "I don't know based on the available documentation."

    #context_text = "\n".join([chunk for _, _, chunk in filtered_chunks])

    # Build the context text from your retrieved chunks
    context_text = ""
    for dist, fn, chunk in top_chunks:
        context_text += f"From {fn}:\n{chunk}\n\n"

    # Define your prompt (system prompt)
    prompt = f"""

You are an assistant that answers questions **strictly** based on the provided context below. 
Follow these rules:
1. If the answer is not explicitly contained in the context, respond with: "I don't know based on the available documentation." and DON'T ANSWER any further.
2. Never mention that you're using provided context.
3. Never use prior knowledge to answer questions.
4. If the question is unrelated to the context, still respond with "I don't know based on the available documentation." and DON'T ANSWER any further.

Context:
{context_text}

User question: {user_question}

Answer:
"""

    # Debug: Print the exact prompt
    print("DEBUG: Final prompt to Ollama:\n", prompt)

    # Finally, call Ollama
    answer = call_ollama(prompt, model=OLLAMA_MODEL)
    print("-----")
    print("Answer:", answer.strip())

if __name__ == "__main__":
    main()