import time
import uuid
from langchain_tavily import TavilyExtract, TavilyMap, TavilyCrawl
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
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


async def index_documents_async(
    store: PineconeVectorStore, documents: list[Document], batch_size: int = 50
):
    texts = [d.page_content for d in documents]
    metadatas = [d.metadata for d in documents]
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    def embed(embeddings, i, batch):
        t0 = time.perf_counter()
        embs = embeddings.embed_documents(batch)
        print(
            f"Store Indexing: Batch {i} - Embedded {len(batch)} docs in {time.perf_counter() - t0:.1f}s"
        )
        return embs

    def upsert(index, i, vectors, batch_size):
        t0 = time.perf_counter()
        index.upsert(vectors=vectors, batch_size=batch_size)
        print(
            f"Store Indexing: Batch {i} - Upsert {batch_size} docs in {time.perf_counter() - t0:.1f}s"
        )

    async def embed_and_upsert(i, batch: list[str]):
        assert store.embeddings is not None
        try:
            embs = await asyncio.to_thread(embed, store.embeddings, i, batch)
            vectors = [
                (str(uuid.uuid4()), emb, {**metadatas[i + j], "text": batch[j]})
                for j, emb in enumerate(embs)
            ]
            await asyncio.to_thread(upsert, store.index, i, vectors, batch_size)
            return True
        except Exception as e:
            print(f"Store indexing: Failed to add batch {i} - {e}")
            return False

    tasks = [embed_and_upsert(i, batch) for i, batch in enumerate(batches)]
    res = await asyncio.gather(*tasks)

    success_count = sum([1 for r in res if r is True])
    if success_count == len(batches):
        print(f"Store indexing: Successfully added all batch")
    else:
        print(
            f"Store indexing: Processed {success_count}/{len(batches)} batches successfully"
        )

    # The code below uses aadd_documents, it has blocking HTTP call to the ollama to forcefully generate embeddings
    # making the performance really bad

    # batches = [
    #     documents[i: i + batch_size] for i in range(0, len(documents), batch_size)
    # ]
    # print(f"Store indexing: documents being indexed through {len(batches)} batches")

    # async def add_batch(batch: list[Document], batch_num: int):
    #     try:
    #         await store.aadd_documents(batch)
    #         print(f"Store indexing: Successfully added batch {batch_num}/{len(batches)} {len(batch)} documents")
    #         return True
    #     except Exception as e:
    #         print(f"Store indexing: Failed to add batch {batch_num} - {e}")
    #         return False

    # tasks = [add_batch(batch, i) for i, batch in enumerate(batches)]
    # results = await asyncio.gather(*tasks, return_exceptions=True)
    # success_count = sum([1 for r in results if r is True])
    # if success_count == len(batches):
    #     print(f"Store indexing: Successfully added all batch")
    # else:
    #     print(f"Store indexing: Processed {success_count}/{len(batches)} batches successfully")


async def async_main():
    index_name = os.getenv("DOCUMENT_ASSISTANT_INDEX_NAME", "")
    dimensions = int(os.getenv("DOCUMENT_ASSISTANT_INDEX_DIMENSIONS", 0))
    if dimensions == 0:
        raise ValueError("Invalid dimensions 0 for index creation")
    model = "qwen3-embedding:4b"
    embeddings = OllamaEmbeddings(model=model, dimensions=dimensions)

    store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    tavily_extract = TavilyExtract()
    tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
    tavily_crawl = TavilyCrawl()

    # Crawl the document site
    res = tavily_crawl.invoke(
        {
            # "url": "https://python.langchain.com",
            # "url": "https://doc.rust-lang.org/stable/book/",
            "url": "https://docs.tavily.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )
    all_docs = [
        Document(page_content=str(r["raw_content"]), metadata={"source": r["url"]})
        for r in res["results"]
    ]
    print(f"Crawl finished, retrieved {len(all_docs)} documents")

    # Splitting the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    print(f"Text splitting finished, docs splitted into {len(splitted_docs)} docs")

    # Indexing the documents batch by batch
    await index_documents_async(store, documents=splitted_docs, batch_size=50)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
