from documents_loader import load_documents
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import PydanticOutputParser
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph,START,END,MessagesState
from typing import TypedDict,Literal
from pydantic import BaseModel,Field


############### Models ##########################

from langchain_ollama import ChatOllama
llm1= ChatOllama(
    model="Qwen3:4b",
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
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embed,
    collection_name="rag_conditional"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
class AgenticRagState(MessagesState):
    query: str
    retrieved_docs: list[Document] | None
    context: str
    response: str
    needs_retrieval: bool


class RouteDecision(BaseModel):
    needs_retrieval: bool = Field(
        description="True if the question requires the specialized documents, otherwise False"
    )

parser = PydanticOutputParser(pydantic_object=RouteDecision)

def route_decision(state: AgenticRagState):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        Decide whether the question requires information from specialized
        documents or can be answered using general knowledge.

        Return True for document-specific questions.
        Return False for general questions.

        {format_instructions}
        """),
        ("human", "{query}")
    ])

    chain = prompt_template | llm1 | parser

    decision = chain.invoke({
        "query": state["query"],
        "format_instructions": parser.get_format_instructions()
    })

    return {"needs_retrieval": decision.needs_retrieval}

def retrive(state: AgenticRagState):
    docs = retriever.invoke(state["query"])
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context,
            "retrieved_docs": docs}
def generate(state: AgenticRagState):
    query = state["query"]
    context = state.get("context") or ""
    if context:
        prompt_template = ChatPromptTemplate.from_messages([
        ("system", "answer the question based on the context provided. If the answer is not present in the context, say I don't know .\n\nContext:\n{context}"),
        ("human", "{query}")
        ])
        chain = prompt_template | llm1
        response = chain.invoke({
            "context" : context,
            "query" : query
        })

    else:
        prompt_template=ChatPromptTemplate.from_messages([
            ("system","answer the following quetion from general knowledge"),
            ("human","{query}")
        ])
        chain = prompt_template | llm1
        response = chain.invoke({"query": query})
    return {"response":response.content}
def route_after_classification(state:AgenticRagState):
    if state["needs_retrieval"]:
        return "retrive"
    return "generate"

############## nodes ##########################

graph = StateGraph(AgenticRagState)
graph.add_node("needs_retrival",route_decision)
graph.add_node("retrive", retrive)
graph.add_node("generate",generate)

graph.add_edge(START, "needs_retrival")

graph.add_conditional_edges(
    "needs_retrival",
    route_after_classification,
    {
        "retrive": "retrive",
        "generate": "generate"
    }
)
graph.add_edge("retrive", "generate")

graph.add_edge("generate", END)





workflow=graph.compile()
workflow

test_query = "what is the evaluation steps for rag application?"

result = workflow.invoke({
    "query": test_query,
    "messages": []
})

print("\n========== ANSWER ==========")
print(result["response"])

# 1. Query
print("\n" + "=" * 60)
print("QUERY")
print("=" * 60)
print(result.get("query", test_query))

# 2. Routing decision
print("\n" + "=" * 60)
print("ROUTING DECISION")
print("=" * 60)
print(f"Needs retrieval: {result.get('needs_retrieval', 'Not available')}")

# 3. Retrieved documents
docs = result.get("retrieved_docs") or result.get("retrived_docs") or []

print("\n" + "=" * 60)
print(f"RETRIEVED DOCUMENTS ({len(docs)})")
print("=" * 60)

for i, doc in enumerate(docs, start=1):
    print(f"\n--- Document {i} ---")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Page: {doc.metadata.get('page', 'Unknown')}")
    print(f"Metadata: {doc.metadata}")
    print(f"Content:\n{doc.page_content}")

# 4. Combined context sent to the LLM
print("\n" + "=" * 60)
print("CONTEXT SENT TO LLM")
print("=" * 60)
print(result.get("context", "No context available"))

# 5. Final answer
print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)
print(result.get("response", "No response available"))

# 6. Total execution time
