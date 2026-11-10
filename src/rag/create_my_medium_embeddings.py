from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

import os

load_dotenv()

def main() -> None:
    index_name = os.getenv("MY_MEDIUM_EMBEDDINGS_INDEX_NAME", "")
    dimensions = int(os.getenv("MY_MEDIUM_EMBEDDINGS_INDEX_DIMENSIONS", 0))
    if dimensions == 0:
        raise ValueError("Invalid dimensions 0 for index creation")
    print(f"Creating index {index_name}")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pc.create_index(
        name=index_name,
        dimension=dimensions,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Index {index_name} created")

if __name__ == "__main__":
    main()
