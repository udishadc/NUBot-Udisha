from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import load_dataset
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from google.cloud.storage import Client

from dataflow.store_data import upload_faiss_index_to_bucket
load_dotenv(override=True)
BUCKET_NAME= os.getenv('BUCKET_NAME')
GOOGLE_APPLICATION_CREDENTIALS=os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
RAW_DATA_FOLDER= os.getenv('RAW_DATA_FOLDER')
def chunk_data():
    # Load all JSON files from a directory
    try:
        storage_client = Client()
        bucket = storage_client.bucket(BUCKET_NAME)
# List files in the bucket
        blobs = bucket.list_blobs(prefix=RAW_DATA_FOLDER)

# Collect the GCS paths for all JSON files (adjust based on your file types)
        gcs_files = [f"gs://{BUCKET_NAME}/{blob.name}" for blob in blobs if blob.name.endswith(".json")]

        dataset = load_dataset("json", data_files=gcs_files,split="train")
        docs = [ Document(
                page_content=item['text'],
                metadata={"url": item['url'], "title": item['title']}
            )for item in dataset if "text" in item] 

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        # Initialize the embedding model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings= HuggingFaceEmbeddings(model_name=model_name,
                                          model_kwargs=model_kwargs,
                                          encode_kwargs=encode_kwargs)
        vector_store = FAISS.from_documents(all_splits, embeddings)
        # Convert documents into FAISS-compatible format
        _ = vector_store.add_documents(documents=all_splits)
        # Save FAISS index
        vector_store.save_local('faiss_index')
        upload_faiss_index_to_bucket()
        return 
    except Exception as e:
        raise Exception(e)


if __name__=="__main__":
    chunk_data()
    upload_faiss_index_to_bucket()
