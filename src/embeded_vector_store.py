from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

from split_document import split_documents_with_metadata

load_dotenv()

def store_vectors(directory, index_name="voice-agent"):
    # Load environment variables

    # Initialize Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key)

    # Create index if needed
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,        # all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")

    index = pc.Index(index_name)

    # Embeddings model
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load and split documents
    split_docs = split_documents_with_metadata(directory)

    vectors = []

    for i, doc in enumerate(split_docs):
        emb = embedding.embed_documents([doc["content"]])[0]

        vectors.append({
            "id": f"chunk_{i}",
            "values": emb,
            "metadata": {
                **doc["metadata"],
                "text": doc["content"]
            }
        })

    # Upsert into Pinecone
    index.upsert(vectors=vectors)

    print(f"Stored {len(vectors)} vectors.")


if __name__ == "__main__":
    directory = "/home/shahanahmed/voice-enabled-AI-agent/documents"
    store_vectors(directory)
