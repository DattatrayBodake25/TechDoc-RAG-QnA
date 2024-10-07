# app.py
from flask import Flask, request, render_template
import faiss
from sentence_transformers import SentenceTransformer
import requests
from data_collection import get_cleaned_documents

app = Flask(__name__)

# Load your FAISS index
faiss_index_path = "document_index.index"  # Renamed for clarity
faiss_index = faiss.read_index(faiss_index_path)  # Renamed variable

# Load the cleaned documents
cleaned_documents = get_cleaned_documents()

def query_index(faiss_index, query, k=5):  # Renamed parameter
    """Queries the FAISS index for the top k nearest neighbors to the given query."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    distances, indices = faiss_index.search(query_embedding, k)  # Updated to use renamed variable
    return distances, indices

def generate_answer(retrieved_docs, query, cohere_api_key):
    """Generates an answer using the Cohere API based on retrieved documents."""
    context = " ".join(retrieved_docs)
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    
    headers = {
        "Authorization": f"Bearer {cohere_api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "prompt": input_text,
        "max_tokens": 150,
        "temperature": 0.7,
    }
    
    response = requests.post("https://api.cohere.ai/generate", headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()['text'].strip()  # Corrected to match API response
    else:
        return f"Error generating answer: {response.text}"

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the index route for the Flask app."""
    answer = ""
    if request.method == 'POST':
        query = request.form['query']
        distances, indices = query_index(faiss_index, query)  # Updated to use renamed variable
        retrieved_docs = [cleaned_documents[idx] for idx in indices[0]]
        cohere_api_key = "my-cohere-api-key"  # Replace with your actual API key
        answer = generate_answer(retrieved_docs, query, cohere_api_key)
    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
