# TechDoc-RAG-QnA

## Overview

TechDoc-RAG-QnA is a Retrieval-Augmented Generation (RAG) system designed to provide quick and accurate answers to queries based on technical documentation. This project utilizes a combination of FAISS for vector similarity search, Sentence Transformers for embedding generation, and the Cohere API for answer generation.

## Features

- **Document Retrieval:** Efficiently retrieve relevant documents using FAISS indexing.
- **Natural Language Processing:** Utilize Sentence Transformers for generating embeddings of the query and documents.
- **Answer Generation:** Generate answers based on the context provided by retrieved documents using the Cohere API.
- **Web Interface:** A simple Flask-based web application for user interaction.

## Technologies Used

- **Python**
- **Flask:** Web framework for building the application.
- **FAISS:** Library for efficient similarity search and clustering of dense vectors.
- **Sentence Transformers:** Pre-trained models for generating sentence embeddings.
- **Cohere API:** API for generating natural language responses.
- **HTML/CSS:** For creating the web interface.

## Project Structure

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/TechDoc-RAG-QnA.git
   cd TechDoc-RAG-QnA
2. Install the required dependencies:
pip install -r requirements.txt
Ensure you have a valid Cohere API key and set it in your code.

Usage
Load the FAISS index by ensuring the document_index.index file is available.

Run the Flask application:
python app.py
Open your web browser and go to http://127.0.0.1:5000/.

Enter your query in the input box and get answers based on the retrieved documents.

Contributing
Contributions are welcome! If you have suggestions for improvements or features, please open an issue or submit a pull request.

Acknowledgments
FAISS for vector similarity search.
Sentence Transformers for embedding generation.
Cohere for natural language generation.
