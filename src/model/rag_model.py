import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess
import json
from src.model.logger import logging
from src.model.exception import CustomException
import sys

# Load the sentence transformer model for embeddings
logging.info("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index & metadata files
INDEX_FILE = 'vector_index.faiss'
METADATA_FILE = 'metadata.pkl'

def load_faiss_index():
    """Load the FAISS index and metadata."""
    try:
        logging.info("Loading FAISS index and metadata...")
        if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
            logging.error("FAISS index or metadata file not found. Please run 'process_html.py' first.")
            raise FileNotFoundError("FAISS index or metadata file not found. Please run 'process_html.py' first.")

        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, 'rb') as f:
            data = pickle.load(f)
        logging.info("FAISS index and metadata loaded successfully.")
        return index, data['documents'], data['metadata']
    except Exception as e:
        raise CustomException(e,sys)

# Load the FAISS index and documents
index, documents, metadata = load_faiss_index()

def retrieve_documents(query, top_k=3):
    """Retrieve the most relevant documents from FAISS index."""
    try:
        logging.info(f"Retrieving top {top_k} relevant documents for query: {query}")
        query_embedding = model.encode([query], convert_to_numpy=True)

        # Search for the nearest documents
        distances, indices = index.search(query_embedding, top_k)
        logging.info("FAISS search completed.")
        
        results = []
        for idx in indices[0]:
            if idx < len(documents):
                results.append(documents[idx])
        
        logging.info(f"Retrieved {len(results)} documents.")
        return results
    except Exception as e:
        raise CustomException(e,sys)

def generate_response(query):
    """Retrieve relevant documents and generate an answer using a local LLM via Ollama."""
    try:
        logging.info(f"Generating response for query: {query}")
        retrieved_docs = retrieve_documents(query)

        # Concatenate retrieved docs as context
        context = "\n".join(retrieved_docs)

        # Prepare the prompt for LLM
        prompt = f"Answer the following question using the given context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        logging.info("Prepared prompt for LLM.")

        # Use Ollama to generate response from a local model like Mistral or Llama2
        ollama_command = ["ollama", "run", "mistral", prompt]
        logging.info("Executing Ollama command...")
        # Ensure UTF-8 decoding to avoid UnicodeDecodeError
        result = subprocess.run(
            ollama_command,
            capture_output=True,
            text=True,
            encoding='utf-8',  # Explicitly set UTF-8 encoding
            errors='replace'  # Replace unsupported characters instead of crashing
        )
        response = result.stdout.strip()
        logging.info("Response generated successfully.")
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        response = f"Error generating response: {e}" 
        raise CustomException(response,sys) 
    return response

if __name__ == '__main__':
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            logging.info("User exited the program.")
            break
        logging.info(f"User query: {query}")
        response = generate_response(query)
        print("\nAnswer:", response, "\n")
