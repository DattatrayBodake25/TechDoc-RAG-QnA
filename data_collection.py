import requests
from bs4 import BeautifulSoup
import re

def fetch_django_docs():
    """Fetches paragraphs from Django documentation."""
    url = "https://docs.djangoproject.com/en/stable/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extract all paragraphs from the documentation
    paragraphs = [p.text for p in soup.find_all('p')]
    return paragraphs

def preprocess_documents(documents):
    """Cleans the fetched documents by removing special characters."""
    cleaned_documents = []
    for doc in documents:
        # Remove special characters, numbers, and excessive whitespace
        doc = re.sub(r'\s+', ' ', doc)  # Replace multiple spaces with a single space
        doc = re.sub(r'[^a-zA-Z\s]', '', doc)  # Remove non-alphabetic characters
        cleaned_documents.append(doc.strip())  # Trim leading/trailing whitespace
    return cleaned_documents

def get_cleaned_documents():
    """Fetches and preprocesses documents, returning cleaned documents."""
    django_docs = fetch_django_docs()
    cleaned_docs = preprocess_documents(django_docs)
    return cleaned_docs

if __name__ == "__main__":
    cleaned_documents = get_cleaned_documents()
    print(f"Cleaned {len(cleaned_documents)} paragraphs from Django documentation.")