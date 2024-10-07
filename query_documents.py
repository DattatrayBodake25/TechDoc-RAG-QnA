import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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

if __name__ == "__main__":
    index_path = "document_index.index"  # Path to your FAISS index
    index = load_index(index_path)

    # Example query
    query = "What is Django?"
    distances, indices = query_index(index, query)
    
    print(f"Query: {query}")
    print(f"Top {len(indices[0])} results:")
    for i, idx in enumerate(indices[0]):
        print(f"Document {idx}: Distance: {distances[0][i]}")