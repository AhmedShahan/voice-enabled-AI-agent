import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key)
index_name = "voice-agent"
index = pc.Index(index_name)

# Initialize embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# LangChain Components
MessageRag = [
    ("system", "You are a Smart AI RAG-based assistant. Answer ONLY from context."),
    ("human", """
    Question: {question}
    Context:
    {context}

    If insufficient info, say: "I don't have enough knowledge based on the document."
    """)
]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)
prompt = ChatPromptTemplate.from_messages(MessageRag)
parser = StrOutputParser()
chain = prompt | llm | parser

# Async RAG function
async def rag_query(query: str):
    # Embed query
    query_emb = embedding.embed_query(query)

    # Retrieve top 10 relevant documents
    result = index.query(vector=query_emb, top_k=10, include_metadata=True)

    # Combine retrieved metadata into context
    context = "\n".join([item['metadata'].get('text', '') for item in result['matches']])

    # Run RAG
    final_answer = await chain.ainvoke({"context": context, "question": query})
    return final_answer

# Async loop
async def main():
    print("Type 'exit' to quit.")
    while True:
        query = input("Enter your question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        answer = await rag_query(query)
        print("\nAnswer:", answer)

# Run the async loop
if __name__ == "__main__":
    asyncio.run(main())