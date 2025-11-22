# workflows/async_embedding_simple.py
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import asyncio

from split_document import split_documents_with_metadata  # your splitter

load_dotenv()

async def store_vectors(directory, index_name="voice-agent"):
    """Asynchronous embedding and storage in Pinecone (simple version)."""
    
    # Initialize Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(api_key)

    # Reuse existing index or create
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        try:
            pc.create_index(
                name=index_name,
                dimension=384,  # for all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"✅ Index '{index_name}' created successfully.")
        except Exception as e:
            print(f"⚠️ Could not create index '{index_name}': {e}")
            if existing_indexes:
                print(f"Using existing index '{existing_indexes[0]}' instead.")
                index_name = existing_indexes[0]
            else:
                raise e
    else:
        print(f"✅ Index '{index_name}' already exists.")

    index = pc.Index(index_name)

    # Initialize embedding model
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ Embedding model initialized.")

    # Load and split documents
    split_docs = split_documents_with_metadata(directory)
    vectors = []

    # Async embedding per document
    for i, doc in enumerate(split_docs):
        emb = await asyncio.to_thread(embedding.embed_documents, [doc["content"]])
        vectors.append({
            "id": f"chunk_{i}",
            "values": emb[0],
            "metadata": {
                **doc["metadata"],
                "text": doc["content"]
            }
        })

    # Upsert into Pinecone
    await asyncio.to_thread(index.upsert, vectors=vectors)
    print(f"✅ Stored {len(vectors)} vectors in index '{index_name}'.")


if __name__ == "__main__":
    directory = "/home/shahanahmed/voice-enabled-AI-agent/documents"
    asyncio.run(store_vectors(directory))
