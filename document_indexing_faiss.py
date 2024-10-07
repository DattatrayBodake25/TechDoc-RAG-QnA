import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from data_collection import get_cleaned_documents

def index_documents(cleaned_documents):
    """Indexes cleaned documents using FAISS."""
    # Load the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for the cleaned documents
    document_embeddings = model.encode(cleaned_documents)

    # Create a FAISS index
    dimension = document_embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)  # Use L2 distance for indexing
    index.add(np.array(document_embeddings))  # Add embeddings to the index

    # Save the index to disk for later use (optional)
    faiss.write_index(index, "document_index.index")

    print(f"Indexed {len(cleaned_documents)} documents with FAISS.")

if __name__ == "__main__":
    cleaned_documents = get_cleaned_documents()  # Get cleaned documents
    index_documents(cleaned_documents)
