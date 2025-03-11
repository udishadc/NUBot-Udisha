from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain.vectorstores import FAISS
import getpass
import os
from dotenv import load_dotenv
import mlflow

mlflow.autolog()
load_dotenv()
if not os.environ.get("MISTRAL_API_KEY"):
  os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")


llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the FAISS index
vector_store = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def generateResponse(query):
# Compile application and test
    try:
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        response = graph.invoke({"question": f"{query}"})
        return response["answer"]
    except Exception as e:
        raise Exception(e)
    
if __name__ == "__main__":
    query=input("generate query")
    generateResponse(query)