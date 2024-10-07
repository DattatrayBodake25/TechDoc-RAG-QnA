import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from data_collection import get_cleaned_documents

def load_index(index_path):
    """Loads the FAISS index from disk."""
    index = faiss.read_index(index_path)
    return index

def query_index(index, query, k=5):
    """Queries the index for the top k nearest neighbors to the given query."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return distances, indices

def generate_answer(retrieved_docs, query, cohere_api_key):
    """Generates an answer using the Cohere API based on retrieved documents."""
    # Combine retrieved documents into context
    context = " ".join(retrieved_docs)
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    
    # Call Cohere API
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
    
    # Print the full response for debugging
    print("API Response:", response.json())
    
    if response.status_code == 200:
        # Update to extract text directly from the response
        answer = response.json()['text'].strip()
        return answer
    else:
        return f"Error generating answer: {response.text}"

if __name__ == "__main__":
    index_path = "document_index.index"
    index = load_index(index_path)

    # Get cleaned documents
    cleaned_documents = get_cleaned_documents()

    # Example query
    query = "What is use of Django?"
    distances, indices = query_index(index, query)

    # Retrieve the relevant documents
    retrieved_docs = [cleaned_documents[idx] for idx in indices[0]]

    # Generate an answer based on the retrieved documents using Cohere API
    cohere_api_key = "my-cohere-api-key"
    answer = generate_answer(retrieved_docs, query, cohere_api_key)
    
    print(f"Query: {query}")
    print(f"Generated Answer: {answer}")
