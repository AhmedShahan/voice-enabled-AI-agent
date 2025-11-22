from dataclasses import dataclass
from typing import Optional

@dataclass
class WorkflowConfig:
    """Configuration for the document ingestion workflow."""
    
    # Document settings
    document_folder_path: str = "/home/shahanahmed/voice-enabled-AI-agent/documents"
    
    # Chunking settings
    chunk_size: int = 2000
    chunk_overlap: int = 500
    
    # Pinecone settings
    pinecone_api_key: Optional[str] = None
    index_name: str = "voice-agent"
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Pinecone index settings
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"