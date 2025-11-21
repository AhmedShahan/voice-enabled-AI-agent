import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("voice-agent")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8)

# RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using the context provided.
If the context doesn't contain relevant information, say "I don't have information about that in my knowledge base." """),
    ("human", "Question: {question}\n\nContext:\n{context}")
])
rag_chain = rag_prompt | llm | StrOutputParser()

def process_query(query: str) -> str:
    """Simple RAG - no classifier, just search and answer."""
    query = query.strip()
    if not query:
        return "Please ask a question."
    
    # Get embeddings and search
    query_emb = embedding.embed_query(query)
    results = index.query(vector=query_emb, top_k=5, include_metadata=True)
    
    # Build context from results
    context = "\n\n".join([
        m['metadata'].get('text', '') 
        for m in results['matches']
    ])
    
    # If no context found
    if not context.strip():
        return "I couldn't find any relevant information in the knowledge base."
    
    # Get answer from LLM
    answer = rag_chain.invoke({"question": query, "context": context})
    return answer.strip()

# CLI test
if __name__ == "__main__":
    print("Ready! Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        print(f"Assistant: {process_query(q)}\n")