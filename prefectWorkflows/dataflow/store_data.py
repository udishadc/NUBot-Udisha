from google.cloud.storage import Client, transfer_manager
import os
from dotenv import load_dotenv
load_dotenv(override=True)
BUCKET_NAME= os.getenv('BUCKET_NAME')
RAW_DATA_FOLDER= os.getenv('RAW_DATA_FOLDER')
FAISS_INDEX_FOLDER= os.getenv('FAISS_INDEX_FOLDER')
GOOGLE_APPLICATION_CREDENTIALS=os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

def get_blob_from_bucket():
    storage_client = Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs()
    for blob in blobs:
    # Get the blob object
        blob = bucket.blob(blob.name)
    # Download the blob content as text (if the file is a text-based file like JSON)
        content = blob.download_as_text()


def upload_many_blobs_with_transfer_manager(
    
):
    """Upload every file in a list to a bucket, concurrently in a process pool.

    Each blob name is derived from the filename, not including the
    `source_directory` parameter. For complete control of the blob name for each
    file (and other aspects of individual blob metadata), use
    transfer_manager.upload_many() instead.
    """

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # A list (or other iterable) of filenames to upload.
    # filenames = ["file_1.txt", "file_2.txt"]

    # The directory on your computer that is the root of all of the files in the
    # list of filenames. This string is prepended (with os.path.join()) to each
    # filename to get the full path to the file. Relative paths and absolute
    # paths are both accepted. This string is not included in the name of the
    # uploaded blob; it is only used to find the source files. An empty string
    # means "the current working directory". Note that this parameter allows
    # directory traversal (e.g. "/", "../") and is not intended for unsanitized
    # end user input.
    # source_directory=""

    # The maximum number of processes to use for the operation. The performance
    # impact of this value depends on the use case, but smaller files usually
    # benefit from a higher number of processes. Each additional process occupies
    # some CPU and memory resources until finished. Threads can be used instead
    # of processes by passing `worker_type=transfer_manager.THREAD`.
    # workers=8

   

    storage_client = Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    source_directory=os.path.join("scraped_data")
    filenames = [f for f in os.listdir(source_directory) if f.endswith(".json")]
    for filename in filenames:
        file_path = os.path.join(source_directory, filename)
        blob = bucket.blob(f"{RAW_DATA_FOLDER}/{filename}")  # Create a blob (object) in the bucket
        print(f"Uploading {filename} to {bucket.name}")
        blob.upload_from_filename(file_path)  # Upload the file



def upload_faiss_index_to_bucket():
    storage_client = Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    source_directory=os.path.join("faiss_index")

    filenames = [f for f in os.listdir(source_directory) ]
    for filename in filenames:
        file_path = os.path.join(source_directory, filename)
        blob = bucket.blob(f"{FAISS_INDEX_FOLDER}/{filename}")  # Create a blob (object) in the bucket
        blob.upload_from_filename(file_path)  # Upload the file

