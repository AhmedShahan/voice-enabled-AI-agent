from ingest_document import ingest_pdf_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents_with_metadata(directory, chunk_size=2000, chunk_overlap=500):
    """
    Load all PDF documents from a directory, split them into chunks while keeping metadata.

    Args:
        directory (str): Path to the directory containing PDF files.
        chunk_size (int, optional): Size of each text chunk. Defaults to 2000.
        chunk_overlap (int, optional): Number of overlapping characters between chunks. Defaults to 500.

    Returns:
        list: List of dictionaries, each with 'content' and 'metadata'.
    """
    # Load documents
    documents = ingest_pdf_documents(directory)

    # Configure text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", " ", ""]
    )

    # Split documents while keeping metadata
    split_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            split_docs.append({
                "content": chunk,
                "metadata": doc.metadata
            })
    
    return split_docs


if __name__ == "__main__":
    directory = "/home/shahanahmed/voice-enabled-AI-agent/documents"
    split_docs = split_documents_with_metadata(directory)

    print("Total Chunks:", len(split_docs))

    # Print first 5 chunks
    for i, chunk in enumerate(split_docs[:5], 1):
        print(f"Chunk {i}:")
        print("Metadata:", chunk["metadata"])
        print("Content:", chunk["content"], "\n")
