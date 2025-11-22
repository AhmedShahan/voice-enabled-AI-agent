# workflows/main_workflow.py
from workflow.document_split import DocumentSplitting
from workflow.vector_embedding import AsyncVectorEmbedding
import os
import asyncio

class AsyncRAGWorkflow:
    def __init__(self, document_folder_path, chunk_size=2000, chunk_overlap=500, 
                 index_name="voice-agent", embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 batch_size=100):
        self.document_folder_path = document_folder_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.batch_size = batch_size
    
    async def run_full_workflow_async(self):
        """Execute the complete RAG pipeline workflow asynchronously."""
        print("Starting ASYNC RAG Pipeline Workflow...")
        print(f"Document folder: {self.document_folder_path}")
        print(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        print(f"Index: {self.index_name}")
        print(f"Embedding model: {self.embedding_model}")
        print(f"Batch size: {self.batch_size} (for async processing)")
        print("-" * 60)
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Step 1: Document Splitting (synchronous - can't be async)
            print("Splitting documents...")
            splitter = DocumentSplitting(
                directory_path=self.document_folder_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            split_docs = splitter.split_documents_with_metadata()
            
            if not split_docs:
                print("No documents to process!")
                return 0
            
            # Step 2: Vector Embedding and Storage (asynchronous)
            print("Embedding and storing vectors (ASYNC)...")
            embedder = AsyncVectorEmbedding(
                index_name=self.index_name,
                embedding_model=self.embedding_model
            )
            
            vector_count = await embedder.store_vectors_async(split_docs, self.batch_size)
            
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            
            print("-" * 60)
            print(f"Workflow completed successfully!")
            print(f"Total chunks processed: {len(split_docs)}")
            print(f"Total vectors stored: {vector_count}")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Speed: {vector_count/processing_time:.2f} vectors/second")
            
            return vector_count
            
        except Exception as e:
            print(f"Error in workflow: {str(e)}")
            raise

def main():
    # Configuration
    DOCUMENT_FOLDER_PATH = "/home/shahanahmed/voice-enabled-AI-agent/documents"
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    INDEX_NAME = "my-custom-index"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    BATCH_SIZE = 50  # Adjust based on your system's capabilities
    
    # Run the async workflow
    workflow = AsyncRAGWorkflow(
        document_folder_path=DOCUMENT_FOLDER_PATH,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        index_name=INDEX_NAME,
        embedding_model=EMBEDDING_MODEL,
        batch_size=BATCH_SIZE
    )
    
    # Run the async workflow
    asyncio.run(workflow.run_full_workflow_async())

if __name__ == "__main__":
    main()