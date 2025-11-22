from langchain_text_splitters import RecursiveCharacterTextSplitter
from workflow.document_ingest import DocumentIngestion

class DocumentSplitting:
    def __init__(self, directory_path, chunk_size=2000, chunk_overlap=500):
        self.directory_path = directory_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ingestor = DocumentIngestion(directory_path)
    
    def split_documents_with_metadata(self):
        """
        Load and split documents into chunks while keeping metadata.
        
        Returns:
            list: List of dictionaries, each with 'content' and 'metadata'.
        """
        # Load documents
        documents = self.ingestor.ingest_pdf_documents()

        # Configure text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
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
        
        print(f"Split documents into {len(split_docs)} chunks")
        return split_docs