import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, ArxivLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# 1. Load Local Documents (Stored Docs)
# Example: load all PDFs from ./docs folder
loader = DirectoryLoader(
    "../docs",                   # path to your folder
    glob="*.pdf",               # can be "*.txt", "*.md", etc.
    loader_cls=PyPDFLoader      # change loader based on file type
)

# loader = ArxivLoader(query="machine learning", max_results=2)
docs = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 3. Embed and store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits)

# 4. Prompt
prompt = hub.pull("rlm/rag-prompt")

# 5. State schema
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# 6. RAG steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    llm = ChatOpenAI(model_name="gpt-4")  # or gpt-3.5-turbo
    response = llm.invoke(messages)
    return {"answer": response.content}

# 7. Compose graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# 8. Example usage
question = "Summarize the key points from the stored documents, describe it document wise."
response = graph.invoke({"question": question})
print("Answer:", response["answer"])
