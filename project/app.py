# app.py
import faiss
import numpy as np
import pickle
import requests
import json
from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# -------------------------------------------------------------------
# 1. Load FAISS index and text chunks
# -------------------------------------------------------------------
index = faiss.read_index("faiss_index.index")
with open("chunks.pkl", "rb") as f:
    all_text_chunks = pickle.load(f)

# Initialize embedding model
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Verify this matches your Ollama model name (run 'ollama list' to check)
OLLAMA_MODEL = "phi4:latest"  # Updated to common default

# -------------------------------------------------------------------
# 2. HTML Template
# -------------------------------------------------------------------
html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DocuBot QA</title>
    <style>
        body { max-width: 800px; margin: 20px auto; padding: 20px; }
        textarea { width: 100%; }
    </style>
</head>
<body>
    <h1>DocuBot QA</h1>
    <form method="POST" action="/ask">
        <label for="question">Ask about the documentation:</label><br>
        <textarea name="question" id="question" rows="4" required></textarea><br>
        <button type="submit">Submit</button>
    </form>
    {% if answer %}
    <div class="answer">
        <h2>Answer:</h2>
        <p>{{ answer }}</p>
        <a href="/">Ask another question</a>
    </div>
    {% endif %}
</body>
</html>
"""

# -------------------------------------------------------------------
# 3. Core Functions
# -------------------------------------------------------------------
def retrieve_chunks(question, top_k=3):
    """Retrieve relevant text chunks from the index"""
    q_embedding = embedder.encode(question).astype('float32').reshape(1, -1)
    distances, indices = index.search(q_embedding, top_k)
    return [(dist, all_text_chunks[idx][0], all_text_chunks[idx][1]) 
            for dist, idx in zip(distances[0], indices[0])]

def call_ollama(prompt, model=OLLAMA_MODEL):
    """Call Ollama API with proper error handling"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.3,  # Lower temp for more focused answers
        "max_tokens": 64
    }
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Ollama connection error: {str(e)}")
        return "Error connecting to AI model"

    generated_text = []
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode('utf-8'))
                generated_text.append(data.get("response", ""))
            except json.JSONDecodeError:
                continue

    return "".join(generated_text).strip()

# -------------------------------------------------------------------
# 4. Flask Routes
# -------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template_string(html_template)

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    if not question:
        return render_template_string(html_template, 
                                    answer="Please enter a question")
    
    # Retrieve and filter chunks
    top_chunks = retrieve_chunks(question)
    MIN_SIMILARITY = 0.7  # Adjust based on your data
    relevant_chunks = [c for c in top_chunks if c[0] <= MIN_SIMILARITY]
    
    if not relevant_chunks:
        return render_template_string(html_template,
                                    answer="I don't know based on the documentation")
    
    # Build context and prompt
    context_text = "\n\n".join([f"From {fn}:\n{chunk}" 
                              for _, fn, chunk in relevant_chunks])
    
    prompt = f"""You are a documentation expert. Answer ONLY using the context below.
If the answer isn't explicitly in the context, say "I don't know."

Context:
{context_text}

Question: {question}

Answer:"""
    
    try:
        answer = call_ollama(prompt)
    except Exception as e:
        app.logger.error(f"Generation error: {str(e)}")
        answer = "Error generating answer"

    return render_template_string(html_template, answer=answer)

# -------------------------------------------------------------------
# 5. Main Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)