from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
import os

class DocumentIngestion:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def ingest_documents(self):
        """
        Load all PDF, Markdown, and Word documents from the specified directory and return them.
        
        Returns:
            list: A list of document objects loaded from the supported files.
        """
        if not os.path.exists(self.directory_path):
            raise ValueError(f"Directory {self.directory_path} does not exist")
        
        loaders = [
            DirectoryLoader(path=self.directory_path, glob="*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(path=self.directory_path, glob="*.md", loader_cls=TextLoader),
            DirectoryLoader(path=self.directory_path, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader),
        ]
        
        all_docs = []
        for loader in loaders:
            docs = loader.load()
            print(f"Loaded {len(docs)} documents from {loader.path} matching {loader.glob}")
            all_docs.extend(docs)
        
        print(f"Total documents loaded: {len(all_docs)}")
        return all_docs
