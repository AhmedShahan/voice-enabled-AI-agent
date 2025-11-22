from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import os

class DocumentIngestion:
    def __init__(self, directory_path):
        self.directory_path = directory_path
    
    def ingest_pdf_documents(self):
        """
        Load all PDF documents from the specified directory and return them.
        
        Returns:
            list: A list of document objects loaded from the PDFs.
        """
        if not os.path.exists(self.directory_path):
            raise ValueError(f"Directory {self.directory_path} does not exist")
            
        loader = DirectoryLoader(
            path=self.directory_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        
        docs = loader.load()
        print(f"Loaded {len(docs)} documents from {self.directory_path}")
        return docs