from langchain_tavily import TavilyExtract, TavilyMap, TavilyCrawl
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
import os
import certifi
import ssl
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio

from dotenv import load_dotenv

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CONTEXT"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

def main():
    index_name = os.getenv("DOCUMENT_ASSISTANT_INDEX_NAME", "")
    dimensions = int(os.getenv("DOCUMENT_ASSISTANT_INDEX_DIMENSIONS", 0))
    if dimensions == 0:
        raise ValueError("Invalid dimensions 0 for index creation")
    model = "qwen3-embedding:8b"
    embeddings = OllamaEmbeddings(model=model, dimensions=dimensions)

    store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    tavily_extract = TavilyExtract()
    tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
    tavily_crawl = TavilyCrawl()


if __name__ == '__main__':
    main()
