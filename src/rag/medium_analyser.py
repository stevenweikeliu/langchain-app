from dotenv import load_dotenv

import os
from operator import itemgetter

# from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough 
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

    
EMBEDDING_MODEL = "qwen3-embedding:8b"
CHAT_MODEL = "qwen3:1.7b"


def format_docs(docs: List[Document]):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    print("Initializing component....")
    dimensions = int(os.getenv("MEDIUM_ANALYSER_INDEX_DIMENSIONS", 0))
    if dimensions == 0:
        raise ValueError("Invalid dimensions 0 for index creation")
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL, dimensions=dimensions)
    index_name = os.getenv("MEDIUM_ANALYSER_INDEX_NAME", "")
    store = PineconeVectorStore(
        embedding = embedding,
        index_name = index_name
    )
    retriever = store.as_retriever()
    
    template = """Answer the question based only on the following context:
{context}
Question: {question}
Provide a detailed answer:"""
    prompt_template = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model=CHAT_MODEL)

    #
    # Implementation 1:  Naive RAG approach without LangChain Expression Language (LCEL)
    #
    def retrieval_chain_without_lcel(query: str):
        """
        simple retrieval chain without LCEL
        """
        docs = retriever.invoke(query)
        context = format_docs(docs)
        messages = prompt_template.format_messages(context=context, question=query)
        response = llm.invoke(messages)
        return response.content


    #
    # Implementation 2:  Better RAG approach with LangChain Expression Language (LCEL)
    #
    def create_retrieval_with_lcel():
        retrieval_chain = (
            RunnablePassthrough.assign(
                context = itemgetter("question") | retriever | format_docs
            )
            | prompt_template
            | llm
            | StrOutputParser()
        )
        return retrieval_chain

    query = "What is Pinecone in machine learning?"

    # 1. Naive implementation
    # res = llm.invoke([HumanMessage(content = query)])
    # print(res.content)

    # 2. RAG implementation without LCEL
    # answer = retrieval_chain_without_lcel(query)

    # 3. RAG implementation with LCEL
    chain = create_retrieval_with_lcel()
    answer = chain.invoke({"question": query})
    print(answer)


if __name__ == '__main__':
    main()
