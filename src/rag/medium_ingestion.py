
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter 
# from langchain_openai import OpenAIEmbeddings
# from unstructured.cleaners.core import clean_extra_whitespace

import os

load_dotenv()

def main():
    print("Ingesting....")
    loader = UnstructuredLoader(
        file_path=f"{os.path.dirname(__file__)}/mediumblog1.txt",
        # post_processors=[clean_extra_whitespace],
        chunking_strategy = "basic",
        max_characters = 1000000,
        encoding = "UTF-8")
    document = loader.load()

    print("splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    # embeddings = OpenAIEmbeddings(openai_api_key = os.getenv("OPENAI_API_KEY"))
    index_name = os.getenv("MY_MEDIUM_EMBEDDINGS_INDEX_NAME", "")
    dimensions = int(os.getenv("MY_MEDIUM_EMBEDDINGS_INDEX_DIMENSIONS", 0))
    if dimensions == 0:
        raise ValueError("Invalid dimensions 0 for index creation")
    model = "qwen3-embedding:8b"
    embeddings = OllamaEmbeddings(model=model, dimensions=dimensions)

    print(f"Persisting embeddings to index {index_name}....")
    PineconeVectorStore.from_documents(texts, embeddings, index_name = index_name)
    print("Finish")

if __name__ == '__main__':
    main()
