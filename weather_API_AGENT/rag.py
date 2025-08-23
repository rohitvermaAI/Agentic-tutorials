# Required installations:
# !pip install langchain langchain-community langchain-text-splitters langgraph langchain-openai

import bs4, os
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# 1. Load Web Page Content
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    ),
)
docs = loader.load()

#suchitra is a nice girl, she is very well in technical aspectes.

##chunk 1 = suchit, chunk 2 = itra i, chunk 3 = is a n, 

# 2. Split Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 3. Embed and Index Chunks in Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # Requires OpenAI API key set
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits) 
 ##link docs is converted into vectors now....

# 4. Define RAG Prompt
prompt = hub.pull("rlm/rag-prompt")

# 5. Define State schema for the application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
# 6. Retrieval and Generation steps
##MAIN components of RAG
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    # Replace this with your preferred chat model (e.g., OpenAI, Google Gemini)
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-4")  # Or "gpt-3.5-turbo", requires OPENAI_API_KEY set
    response = llm.invoke(messages)
    return {"answer": response.content}

# 7. Compose the chain as a graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# 8. Example usage
question = "What is Task Decomposition?"## {"question": "What is Task Decomposition?"}
response = graph.invoke({"question": question})
print("Answer:", response["answer"])


##graph: retrieve(question) -context->generate