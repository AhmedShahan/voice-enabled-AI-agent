# run_async_workflow.py
"""
Async workflow runner for the RAG pipeline.
This file runs the workflow asynchronously for maximum speed.
"""

from workflow.main_workflow import AsyncRAGWorkflow
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def run_custom_async_workflow():
    """Run a custom async RAG workflow with specific parameters."""
    
    # Define your custom configuration
    config = {
        "document_folder_path": "/home/shahanahmed/voice-enabled-AI-agent/documents",
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "index_name": "my-custom-index",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 100  # Larger batch size for better async performance
    }
    
    print("⚡🚀 Running CUSTOM ASYNC RAG Workflow")
    print("=" * 60)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    try:
        # Initialize and run the async workflow
        workflow = AsyncRAGWorkflow(**config)
        vector_count = await workflow.run_full_workflow_async()
        
        print(f"\n✅ Custom async workflow completed successfully!")
        print(f"🎯 Stored {vector_count} vectors in the knowledge base!")
        
    except Exception as e:
        print(f"\n❌ Error running custom async workflow: {str(e)}")
        raise

async def run_performance_comparison():
    """Run multiple configurations to compare performance."""
    
    configurations = [
        {
            "document_folder_path": "/home/shahanahmed/voice-enabled-AI-agent/documents",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "index_name": "performance-test-small",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 50
        },
        {
            "document_folder_path": "/home/shahanahmed/voice-enabled-AI-agent/documents",
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "index_name": "performance-test-medium",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 100
        },
        {
            "document_folder_path": "/home/shahanahmed/voice-enabled-AI-agent/documents",
            "chunk_size": 2000,
            "chunk_overlap": 400,
            "index_name": "performance-test-large",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 150
        }
    ]
    
    print("⚡📊 Running Performance Comparison")
    print("=" * 70)
    
    results = []
    
    for i, config in enumerate(configurations, 1):
        print(f"\n🔄 Testing configuration {i}/{len(configurations)}")
        print(f"   Index: {config['index_name']}")
        print(f"   Chunk: {config['chunk_size']}, Overlap: {config['chunk_overlap']}")
        print(f"   Batch: {config['batch_size']}")
        print("-" * 70)
        
        try:
            workflow = AsyncRAGWorkflow(**config)
            start_time = asyncio.get_event_loop().time()
            vector_count = await workflow.run_full_workflow_async()
            end_time = asyncio.get_event_loop().time()
            
            processing_time = end_time - start_time
            speed = vector_count / processing_time if processing_time > 0 else 0
            
            results.append({
                "config": config["index_name"],
                "vectors": vector_count,
                "time": processing_time,
                "speed": speed
            })
            
            print(f"✅ Configuration {i} completed in {processing_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Configuration {i} failed: {str(e)}")
            continue
    
    # Print summary
    if results:
        print("\n" + "="*70)
        print(" PERFORMANCE SUMMARY ")
        print("="*70)
        for result in results:
            print(f"{result['config']}: {result['speed']:.2f} vectors/sec")
        
        best = max(results, key=lambda x: x['speed'])
        print(f"\n🏆 Fastest configuration: {best['config']} "
              f"({best['speed']:.2f} vectors/sec)")

if __name__ == "__main__":
    # Run single custom async workflow
    asyncio.run(run_custom_async_workflow())
    
    # Uncomment to run performance comparison
    # asyncio.run(run_performance_comparison())