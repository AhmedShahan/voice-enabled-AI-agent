import asyncio
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize everything (same as before)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("voice-agent")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8)  # slightly lower temp = more consistent

# ————————————————————————
# 1. Smart Greeting + Off-topic Detection Chain (uses LLM intelligence)
# ————————————————————————
greeting_classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an extremely smart conversation classifier.
Your only job is to look at the user's message and return exactly ONE of these words (nothing else):

GREETING    → if it's a casual hello, how are you, who are you, good morning, etc.
DOCUMENT    → if the question is clearly trying to get information from the knowledge base/document
OFFTOPIC    → anything else (weather, jokes, opinions, math, etc.)

Examples:
"hi" → GREETING
"what is the capital of France" → OFFTOPIC
"tell me about the voice agent features" → DOCUMENT
"who created you?" → GREETING
"how are you today" → GREETING
"""),
    ("human", "{query}")
])

classifier_chain = greeting_classifier_prompt | llm | StrOutputParser()

# ————————————————————————
# 2. Friendly Casual Response Chain (fully intelligent & varied every time)
# ————————————————————————
casual_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a super friendly, warm, and slightly playful AI assistant.
Answer in 1-2 short sentences max. Use emojis sparingly but nicely.
Never mention "document" unless the user already did.
Keep it extremely natural and human-like."""),
    ("human", "{query}")
])

casual_chain = casual_prompt | llm | StrOutputParser()

# ————————————————————————
# 3. Strict RAG Chain (exactly like before, but stricter)
# ————————————————————————
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a precise RAG assistant. 
Answer ONLY using the retrieved context below. 
If the answer is not in context → reply exactly: "I don't have enough information in the document to answer that." 
Do NOT make anything up. Do NOT be friendly here — be professional and direct."""),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

rag_chain = rag_prompt | llm | StrOutputParser()

# ————————————————————————
# Main RAG function
# ————————————————————————
async def rag_query(query: str) -> str:
    query_emb = embedding.embed_query(query)
    results = index.query(vector=query_emb, top_k=8, include_metadata=True)
    context = "\n\n".join([m['metadata'].get('text', '') for m in results['matches']])
    answer = await rag_chain.ainvoke({"question": query, "context": context})
    return answer.strip()

# ————————————————————————
# Main decision logic
# ————————————————————————
async def process_query(query: str) -> str:
    query = query.strip()
    if not query:
        return "??"

    # Step 1: Classify intent with LLM (super accurate & zero hardcoding)
    classification = await classifier_chain.ainvoke({"query": query})
    classification = re.sub(r"[^A-Z]", "", classification.upper())[:10]  # clean noise

    if "GREETING" in classification:
        return await casual_chain.ainvoke({"query": query})

    elif "DOCUMENT" in classification:
        answer = await rag_query(query)
        if "don't have enough information" in answer.lower():
            return ("I'm sorry, I couldn't find that in the document.\n"
                    "This knowledge base is about the Voice Agent system (features, architecture, usage, etc.).\n"
                    "Would you like to ask something related to that?")
        return answer

    else:  # OFFTOPIC
        return ("That's a fun question, but I'm specialized only in the uploaded document (Voice Agent system).\n"
                "Ask me anything about its features, setup, API, or how it works — I'm really good at that!")

# ————————————————————————
# Chat loop
# ————————————————————————
async def main():
    print("Voice Agent Assistant is ready! (type 'exit' to quit)\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit", "bye"]:
            print("Assistant: Bye-bye! Come back anytime")
            break

        response = await process_query(query)
        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())