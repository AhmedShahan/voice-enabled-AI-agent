from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
# Load .env
load_dotenv()
# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key)

# print(pc.list_indexes().names())

index_name = "voice-agent"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 outputs 384-dim vectors
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Database created Successully") 
else:
    print(f"{index_name} is already exists")

index = pc.Index(index_name)
# # Embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectors = []


from split_document import split_documents_with_metadata

split_docs=split_documents_with_metadata("/home/shahanahmed/voice-enabled-AI-agent/documents")
for i, doc in enumerate(split_docs):
    # Embed the chunk text
    emb = embedding.embed_documents([doc["content"]])[0]

    vectors.append({
        "id": f"chunk_{i}",
        "values": emb,
        "metadata": {
            **doc["metadata"],        # keep original metadata
            "text": doc["content"]    # store chunk text
        }
    })
# Upsert into Pinecone
index.upsert(vectors=vectors, namespace="BD")