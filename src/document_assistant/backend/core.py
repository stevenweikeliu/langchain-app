from langchain.messages import ToolMessage
from typing import Tuple
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document

load_dotenv()

index_name = os.getenv("DOCUMENT_ASSISTANT_INDEX_NAME", "")
dimensions = int(os.getenv("DOCUMENT_ASSISTANT_INDEX_DIMENSIONS", 0))
if dimensions == 0:
    raise ValueError("Invalid dimensions 0 for index creation")
model = "qwen3-embedding:4b"
embeddings = OllamaEmbeddings(model=model, dimensions=dimensions)

store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

chat_model = init_chat_model(model="qwen3:1.7b", model_provider="ollama")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> Tuple[str, list[Document]]:
    """
    Retrieves the context/documentation for the given query from the RAG vector store
    Args:
        query - The question from the user
    Returns:
        tuple of the concatenated context/documentation content with '\n\n' and the list of documents retrieved from vector store
    """
    retrieved_docs = store.as_retriever().invoke(query, k=4)
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        )
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def make_agent():
    tools = [retrieve_context]
    query = "Tell me about Rust programming language?"
    system_prompt = (
        "You are a helpful AI assistant that answers questions about technology."
        "You have access to a tool that retrieves context from relevant documentations."
        "Use the tool to find relevant context before answering questions."
        "Always cite the source you use in your answers."
        "If you cannot find the answer from the retrieved context, say so."
    )
    agent = create_agent(chat_model, tools=tools, system_prompt=system_prompt)
    return agent


def ask_agent(agent, query):
    messages = [{"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})
    answer = response["messages"][-1].content
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    return {"answer": answer, "context": context_docs}


def main() -> None:
    agent = make_agent()

    while True:
        query = input("Ask here >>> ")
        res = ask_agent(agent, query)
        print(res.get("answer"))
        print(
            "\n\n".join(
                [
                    f"source: {doc.metadata.get('source', 'Unknown')}"
                    for doc in res.get("context")
                ]
            )
        )


if __name__ == "__main__":
    main()
