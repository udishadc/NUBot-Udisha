from functools import lru_cache
# from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import getpass
import os
from dotenv import load_dotenv
import mlflow
import time
from langfair.auto import AutoEval
import asyncio
# Load the FAISS index
from google.cloud.storage import Client
import tempfile
import os
load_dotenv(override=True)
mlflow.langchain.autolog()
MLFLOW_TRACKING_URI =os.environ.get("MLFLOW_TRACKING_URI")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
FAISS_INDEX_FOLDER= os.getenv('FAISS_INDEX_FOLDER')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # Remote MLflow Server
# Where you currently have this line:
mlflow.set_experiment("rag_experiment")
def get_or_create_experiment(experiment_name):

    # Check if experiment exists
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is not None:
            # Check if experiment is active (not deleted)
            if experiment.lifecycle_stage == "active":
                print(f"Found active experiment '{experiment_name}' with ID: {experiment.experiment_id}")
                return experiment.experiment_id
            else:
                # Experiment exists but is deleted, create a new one with timestamp
                new_name = f"{experiment_name}_{int(time.time())}"
                experiment_id = mlflow.create_experiment(new_name)
                print(f"Original experiment was deleted. Created new experiment '{new_name}' with ID: {experiment_id}")
                return experiment_id
        else:
            # Create new experiment
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
            return experiment_id
    except Exception as e:
        print(f"Error getting or creating experiment: {e}")
        # Fallback - create a new experiment with timestamp
        new_name = f"{experiment_name}_{int(time.time())}"
        experiment_id = mlflow.create_experiment(new_name)
        print(f"Created fallback experiment '{new_name}' with ID: {experiment_id}")
        return experiment_id

# Replace it with:
experiment_id = get_or_create_experiment("rag_experiment")
mlflow.set_experiment_tag("description", "RAG pipeline with Mistral AI model")
if not os.environ.get("MISTRAL_API_KEY"):
  os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

@lru_cache(maxsize=None)
def get_llm():
    llm = init_chat_model("mistral-tiny-latest", model_provider="mistralai")
    return llm

@lru_cache(maxsize=None)
def get_prompt():
# Define prompt for question-answering
    # Your prompt template
    template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    return custom_rag_prompt

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


@lru_cache(maxsize=None)
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings



# Initialize GCS client
storage_client = Client()
bucket=storage_client.bucket(os.getenv('BUCKET_NAME'))
embeddings=load_embeddings()
if not os.path.exists(FAISS_INDEX_FOLDER):
    os.makedirs(FAISS_INDEX_FOLDER, exist_ok=True)
# Create a temporary directory
# Download FAISS index files from bucket to FAISS_INDEX_FOLDER directory
for blob in bucket.list_blobs(prefix=FAISS_INDEX_FOLDER):
    # Extract just the filename from the full path
    filename = os.path.basename(blob.name)
    local_path = os.path.join(FAISS_INDEX_FOLDER, filename)
    blob.download_to_filename(local_path)

# Load FAISS index from directory
vector_store = FAISS.load_local(FAISS_INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)
# Define application steps
def retrieve(state: State):
    with mlflow.start_run(nested=True, run_name="retrieval",experiment_id=experiment_id):
        start_time = time.time()
        retrieved_docs = vector_store.similarity_search(state["question"])
        retrieval_time = time.time() - start_time
    
        # Extract only metadata
        doc_metadata = [{"doc_id": doc.metadata.get("id", i), "source": doc.metadata.get("source", "unknown")}
                        for i, doc in enumerate(retrieved_docs)]
        
        # Log metadata instead of full documents
        mlflow.log_metric("retrieval_time", retrieval_time)
        mlflow.log_param("retrieved_docs_count", len(retrieved_docs))
        mlflow.log_dict(doc_metadata, "retrieved_docs.json")

    return {"context": retrieved_docs}

# Initialize LLM once and store in a global variable
llm = get_llm()
# Initialize prompt once and store in a global variable
prompt = get_prompt()
def generate(state: State):
    with mlflow.start_run(nested=True, run_name="generation",experiment_id=experiment_id):
        start_time = time.time()
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        token_count = len(docs_content.split()) 
        # Use the global prompt instance
        mlflow.log_param("retrieved_tokens", token_count)
        mlflow.log_param("context_length", len(docs_content))
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        generation_time = time.time() - start_time
        
        # Log LLM generation performance
        mlflow.log_metric("generation_time", generation_time)
        mlflow.log_param("response_length", len(response.content.split()))
        mlflow.log_param("model_name", "mistral-tiny-latest")

        # Save response
        # with open("response.txt", "w") as f:
        #     f.write(response.content)
        # mlflow.log_artifact("response.txt")

    return {"answer": response.content}


def generateResponse(query):
# Compile application and test
    try:
         with mlflow.start_run(run_name="RAG_Pipeline",experiment_id=experiment_id):
            mlflow.log_param("query", query)
            graph_builder = StateGraph(State).add_sequence([retrieve, generate])
            graph_builder.add_edge(START, "retrieve")
            graph = graph_builder.compile()
            response = graph.invoke({"question": f"{query}"})
            mlflow.log_param("final_answer", response["answer"])
            return response["answer"]
    except Exception as e:
        mlflow.log_param("error", str(e))
        raise Exception(e)
    
async def checkModel_fairness():
    auto_object = AutoEval(
        prompts=["tell me about khoury college"], 
        langchain_llm=llm,
        # toxicity_device=device # uncomment if GPU is available
    )
    results = await auto_object.evaluate()
    print(results['metrics'])
    
if __name__ == "__main__":

    query=input("generate query")
    response=generateResponse(query)
    print(response)
    #uncomment and enter prompts for model fairness and there is a limitation on api key
    # asyncio.run(checkModel_fairness())