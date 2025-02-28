import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "scraped_data"
INDEX_FILE = "vector_index.faiss"
METADATA_FILE = "metadata.pkl"

def load_json_files():
    """Load all JSON files and extract relevant data."""
    file_paths = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith(".json")]
    
    documents = []
    metadata = []

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                documents.append(data.get("text", ""))
                metadata.append({
                    "url": data.get("url", "No URL"),
                    "title": data.get("title", "No Title")
                })
            else:
                print(f"Skipping file {file_path} as it does not contain a dictionary at the top level.")
    
    return documents, metadata

def create_vector_database(documents):
    """Compute embeddings for documents and create a FAISS index."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents, convert_to_numpy=True)

    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    return index, embeddings

def trainModel():
    """Main function to process scraped JSON data."""
    documents, metadata = load_json_files()

    index, _ = create_vector_database(documents)

    with open(METADATA_FILE, 'wb') as f:
        pickle.dump({"metadata": metadata, "documents": documents}, f)

    print("Processing complete. FAISS index and metadata saved.")

if __name__ == "__main__":
    trainModel()