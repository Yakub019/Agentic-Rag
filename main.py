import sentence_transformers
from documents_loader import load_documents
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma
from langgraph.graph import StateGraph,START,END,MessagesState
from typing import TypedDict
from pydantic import BaseModel


############### Models ##########################

from langchain_ollama import ChatOllama
llm1= ChatOllama(
    model="Qwen3:1.7b",
    temperature=0,
)
llm2 = HuggingFaceEndpoint(
    repo_id="Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0
)
model = ChatHuggingFace(llm=llm2)

embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

############Splitter ##########################

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150)
chunks=splitter.split_documents(load_documents())
print(f"Number of chunks: {len(chunks)}")

############ Vector Store and retriver ##########################

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embed,
    collection_name="rag-base")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

'''print("Retriever is ready to use.")

docs=retriever.invoke("What is the rag triad")
print("\n\n".join([doc.page_content for doc in docs]))'''

################ State Graph ##########################

class AgenticRagState(MessagesState):
    query: str
    retrieved_docs: list[str]
    context: str
    response: str


############## Functions ##########################
def retrive(state: AgenticRagState):
    docs = retriever.invoke(state["query"])
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context,
            "retrieved_docs": docs}
def generate(state: AgenticRagState):
    query = state["query"]
    context = state["context"]
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "answer the question based on the context provided. If the answer is not present in the context, say I don't know .\n\nContext:\n{context}"),
        ("human", "{query}")
        ])
    response = (prompt_template|llm1).invoke({"context": context, "query": query})
    return {"response": response}

############## nodes ##########################

graph = StateGraph(AgenticRagState)
graph.add_node("retrive", retrive)
graph.add_node("generate",generate)
graph.add_edge(START, "retrive")
graph.add_edge("retrive", "generate")
graph.add_edge("generate", END)
workflow=graph.compile()


test_query = "What is the Regression Testing?"

result = workflow.invoke({
    "query": test_query,
    "messages": []
})

print("\n========== ANSWER ==========")
print(result["response"].content)

print("\n========== RETRIEVED DOCUMENTS ==========")

for i, doc in enumerate(result["retrieved_docs"], start=1):
    print(f"\n--- Document {i} ---")
    print("Metadata:", doc.metadata)
    print("Content:")
    print(doc.page_content)
    print("-" * 60)

