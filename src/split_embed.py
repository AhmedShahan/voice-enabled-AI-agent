from ingest_document import ingest_pdf_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
directory = "/home/shahanahmed/voice-enabled-AI-agent/documents"
documents = ingest_pdf_documents(directory)


# Configure text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,       # Increase chunk_size for meaningful chunks
    chunk_overlap=500,
    separators=["\n", " ", ""]
)

# Split each document while keeping metadata
split_docs = []
for doc in documents:
    chunks = splitter.split_text(doc.page_content)
    for chunk in chunks:
        split_docs.append({
            "content": chunk,
            "metadata": doc.metadata
        })

# Example: print first 5 chunks
# for i, chunk in enumerate(split_docs[:5], 1):
#     print(f"Chunk {i}:")
#     print("Metadata:", chunk["metadata"])
#     print("Content:", chunk["content"], "\n")


print("Total Chunks: ", len(split_docs))
# Example: print first 5 chunks
for i, chunk in enumerate(split_docs[:5], 1):
    print(f"Chunk {i}:")
    print("Metadata:", chunk["metadata"])
    print("Content:", chunk["content"], "\n")
