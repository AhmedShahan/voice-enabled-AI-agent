from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def load_pdf_documents(directory_path):
    """
    Load all PDF documents from the specified directory and return them.
    
    Args:
        directory_path (str): Path to the directory containing PDF files.
        
    Returns:
        list: A list of document objects loaded from the PDFs.
    """
    loader = DirectoryLoader(
        path=directory_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    docs = loader.load()
    return docs

if __name__ == "__main__":
    directory = "/home/shahanahmed/voice-enabled-AI-agent/documents"
    documents = load_pdf_documents(directory)

    for doc in documents:
        print("Metadata:", doc.metadata)
        print("Content Preview:", doc.page_content[:200], "\n")  # First 200 characters
