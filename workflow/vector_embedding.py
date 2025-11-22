# workflows/vector_embedding.py
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

class AsyncVectorEmbedding:
    def __init__(self, index_name="voice-agent", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = None

    def initialize_pinecone(self):
        """Initialize Pinecone connection and reuse an existing index, or create if possible."""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        pc = Pinecone(api_key=api_key)

        # Always try to reuse existing index
        existing_indexes = pc.list_indexes().names()
        if self.index_name in existing_indexes:
            print(f"✅ Index '{self.index_name}' already exists, reusing it.")
        else:
            try:
                print(f"Index '{self.index_name}' not found. Attempting to create...")
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print(f"✅ Index '{self.index_name}' created successfully.")
            except Exception as e:
                # Likely a 403 due to max serverless indexes
                print(f"⚠️ Could not create index '{self.index_name}': {str(e)}")
                print("Using existing index (must reuse an existing one)")

        self.index = pc.Index(self.index_name)
        return self.index

    def initialize_embeddings(self):
        """Initialize the embedding model."""
        embedding = HuggingFaceEmbeddings(
            model_name=self.embedding_model
        )
        print(f"✅ Embedding model '{self.embedding_model}' initialized")
        return embedding

    async def generate_embeddings_async(self, texts, embedding_model):
        """Generate embeddings asynchronously using thread pool."""
        def embed_batch(text_batch):
            return embedding_model.embed_documents(text_batch)

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            results = await loop.run_in_executor(executor, embed_batch, texts)
        return results

    async def store_vectors_async(self, split_docs, batch_size=100):
        """Store document chunks as vectors in Pinecone asynchronously using namespaces."""
        index = self.initialize_pinecone()
        embedding_model = self.initialize_embeddings()

        total_vectors = len(split_docs)
        vectors_stored = 0

        print(f"🚀 Starting async embedding generation and storage...")
        print(f"📊 Total chunks to process: {total_vectors}")
        print(f"📦 Batch size: {batch_size}")

        # Process in batches
        for i in range(0, total_vectors, batch_size):
            batch_docs = split_docs[i:i + batch_size]
            # namespace = f"batch_{i//batch_size + 1}"  # Use namespace per batch

            print(f"🔄 Processing batch {i//batch_size + 1} ({len(batch_docs)} chunks)...")

            texts = [doc["content"] for doc in batch_docs]

            try:
                embeddings = await self.generate_embeddings_async(texts, embedding_model)

                vectors = []
                for j, (doc, emb) in enumerate(zip(batch_docs, embeddings)):
                    chunk_id = f"chunk_{i+j}"
                    vectors.append({
                        "id": chunk_id,
                        "values": emb,
                        "metadata": {
                            **doc["metadata"],
                            "text": doc["content"],
                            # "batch": i//batch_size + 1,
                            # "position_in_batch": j
                        }
                    })

                # Upsert using namespace
                index.upsert(vectors=vectors)
                vectors_stored += len(vectors)

                print(f"✅ Batch {i//batch_size + 1} completed. Upserted {len(vectors)} vectors.")
            except Exception as e:
                print(f"❌ Error processing batch {i//batch_size + 1}: {str(e)}")
                continue

        print(f"✅ Async storage completed! Stored {vectors_stored}/{total_vectors} vectors.")
        return vectors_stored
