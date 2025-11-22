# workflow_execution.py
"""
Main entry point for Docker container workflow execution.
This script reads configuration from environment variables and config.yaml,
then executes the RAG pipeline workflow.
"""

import os
import asyncio
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv
from workflow.main_workflow import AsyncRAGWorkflow

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_file="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Configuration loaded from {config_file}")
        return config
    except Exception as e:
        logger.warning(f"⚠️  Could not load config file: {e}. Using environment variables only.")
        return {}

def get_workflow_config():
    """
    Get workflow configuration from environment variables or config file.
    Environment variables take precedence over config file.
    """
    config_file = load_config()
    
    # Get configuration with priority: ENV > config.yaml > default
    document_folder_path = os.getenv(
        "DOCUMENT_FOLDER_PATH",
        config_file.get("document_processing", {}).get("folder_path", "/app/documents")
    )
    
    chunk_size = int(os.getenv(
        "CHUNK_SIZE",
        config_file.get("document_processing", {}).get("chunk_size", 1500)
    ))
    
    chunk_overlap = int(os.getenv(
        "CHUNK_OVERLAP",
        config_file.get("document_processing", {}).get("chunk_overlap", 300)
    ))
    
    index_name = os.getenv(
        "INDEX_NAME",
        config_file.get("vector_store", {}).get("index_name", "my-custom-index")
    )
    
    embedding_model = os.getenv(
        "EMBEDDING_MODEL",
        config_file.get("vector_store", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    )
    
    batch_size = int(os.getenv(
        "BATCH_SIZE",
        config_file.get("processing", {}).get("batch_size", 100)
    ))
    
    return {
        "document_folder_path": document_folder_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "index_name": index_name,
        "embedding_model": embedding_model,
        "batch_size": batch_size
    }

def validate_environment():
    """Validate required environment variables and paths."""
    errors = []
    
    # Check for required API keys
    if not os.getenv("PINECONE_API_KEY"):
        errors.append("PINECONE_API_KEY is not set")
    
    # Check if document folder exists and has files
    config = get_workflow_config()
    doc_path = Path(config["document_folder_path"])
    
    if not doc_path.exists():
        errors.append(f"Document folder does not exist: {doc_path}")
    elif not list(doc_path.glob("*.pdf")):
        errors.append(f"No PDF files found in: {doc_path}")
    
    if errors:
        logger.error("❌ Validation failed:")
        for error in errors:
            logger.error(f"   - {error}")
        return False
    
    logger.info("✅ Environment validation passed")
    return True

async def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("🚀 RAG WORKFLOW EXECUTION - DOCKER CONTAINER")
    logger.info("=" * 70)
    
    # Validate environment
    if not validate_environment():
        logger.error("❌ Exiting due to validation errors")
        return 1
    
    # Get configuration
    config = get_workflow_config()
    
    logger.info("📋 Workflow Configuration:")
    for key, value in config.items():
        logger.info(f"   {key}: {value}")
    logger.info("=" * 70)
    
    try:
        # Initialize workflow
        workflow = AsyncRAGWorkflow(**config)
        
        # Execute workflow
        vector_count = await workflow.run_full_workflow_async()
        
        logger.info("=" * 70)
        logger.info(f"✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 Total vectors stored: {vector_count}")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"❌ WORKFLOW FAILED: {str(e)}")
        logger.error("=" * 70)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)