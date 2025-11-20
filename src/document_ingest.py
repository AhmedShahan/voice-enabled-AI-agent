from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loder=DirectoryLoader(
  
    path="/home/shahanahmed/voice-enabled-AI-agent/documents",
    glob="*.pdf",
    ## direcory থেকে সব pdf file loadd করা
    loader_cls=PyPDFLoader
)


docs=loder.load()

# print(len(docs))
#  
for document in docs:
    print("Page Content\n")
    print(document.metadata)