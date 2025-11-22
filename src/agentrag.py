# agentrag_async.py
"""
Smart Query Router with Weather Tool Integration and RAG Pipeline (async)
Minimal changes from original code; faster responses while keeping context.
"""

import os
import requests
import time
import asyncio
from typing import Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# =============================================================================
# INITIALIZE COMPONENTS
# =============================================================================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("my-custom-index")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Cache embeddings to reduce repeated Pinecone queries
embedding_cache = {}

async def get_query_embedding(query: str):
    if query in embedding_cache:
        return embedding_cache[query]
    emb = embedding.embed_query(query)
    embedding_cache[query] = emb
    return emb

# =============================================================================
# WEATHER TOOL
# =============================================================================
@tool
def get_weather(city: str) -> str:
    """
    Get current weather information for a given city.
    Use this tool when the user asks about weather, temperature, or climate conditions.
    """
    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {'key': WEATHER_API_KEY, 'q': city, 'days': 1, 'aqi': 'no'}
    try:
        response = requests.get(url, params=params, timeout=10).json()
        if response.get('error'):
            return f"Sorry, couldn't find weather for '{city}': {response['error']['message']}"
        current = response['current']
        astro = response['forecast']['forecastday'][0]['astro']
        location = response['location']['name']
        country = response['location']['country']
        return f"""Weather for {location}, {country}:
Temperature: {current['temp_c']}°C (feels like {current['feelslike_c']}°C)
Humidity: {current['humidity']}%
Wind: {current['wind_kph']} km/h {current['wind_dir']}
Condition: {current['condition']['text']}
Sunrise: {astro['sunrise']}
Sunset: {astro['sunset']}"""
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

# =============================================================================
# DOCUMENT TOPICS
# =============================================================================
available_topics_cache = []

async def get_available_topics() -> list:
    global available_topics_cache
    if available_topics_cache:
        return available_topics_cache
    try:
        dummy_query = await get_query_embedding("document")
        results = index.query(vector=dummy_query, top_k=50, include_metadata=True)
        sources = set()
        for match in results['matches']:
            source = match['metadata'].get('source', '')
            if source:
                filename = os.path.basename(source)
                sources.add(filename)
        available_topics_cache = list(sources)
        return available_topics_cache
    except Exception:
        return []

# =============================================================================
# QUERY CLASSIFIER
# =============================================================================
async def classify_query(query: str) -> str:
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query classifier. Classify the user's query into exactly one category:
1. "weather" - If asking about weather, temperature, climate, sunrise, sunset for any location
2. "document" - If asking questions that need information from documents/knowledge base
3. "general" - If it's a greeting, chitchat, or unclear query
Respond with ONLY one word: weather, document, or general"""),
        ("human", "{query}")
    ])
    classifier_chain = classifier_prompt | llm | StrOutputParser()
    result = classifier_chain.invoke({"query": query}).strip().lower()
    if "weather" in result:
        return "weather"
    elif "document" in result:
        return "document"
    else:
        return "general"

# =============================================================================
# RAG PIPELINE
# =============================================================================
async def rag_search(query: str) -> Tuple[str, float]:
    query_emb = await get_query_embedding(query)
    results = index.query(vector=query_emb, top_k=5, include_metadata=True)
    if not results['matches']:
        return "", 0.0
    top_score = max(m['score'] for m in results['matches'])
    context = "\n\n---\n\n".join([
        f"[Source: {m['metadata'].get('source', 'Unknown')}]\n{m['metadata'].get('text', '')}"
        for m in results['matches']
    ])
    return context, top_score

async def generate_rag_response(query: str, context: str) -> str:
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly. Be concise and accurate."""),
        ("human", """Question: {question}

Context:
{context}

Answer:""")
    ])
    rag_chain = rag_prompt | llm | StrOutputParser()
    return rag_chain.invoke({"question": query, "context": context})

# =============================================================================
# WEATHER AGENT
# =============================================================================
tools = [get_weather]
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
weather_agent = AgentExecutor(agent=agent, tools=tools, verbose=False)

async def handle_weather_query(query: str) -> str:
    try:
        response = weather_agent.invoke({"input": query})
        return response['output']
    except Exception as e:
        return f"Sorry, I couldn't fetch the weather information. Error: {str(e)}"

# =============================================================================
# FRIENDLY RESPONSE
# =============================================================================
async def generate_friendly_response(query: str) -> str:
    topics = await get_available_topics()
    topics_str = ", ".join(topics[:5]) if topics else "various topics"
    friendly_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a friendly AI assistant. The user's query doesn't match your knowledge base.
Available document topics: {topics_str}
Respond warmly and:
1. Acknowledge you don't have specific information for their query
2. Mention you have documents about: {topics_str}
3. Ask if they'd like to know something about those topics
4. You can also help with weather information for any city
Keep response concise and helpful."""),
        ("human", "{query}")
    ])
    friendly_chain = friendly_prompt | llm | StrOutputParser()
    return friendly_chain.invoke({"query": query})

# =============================================================================
# MAIN QUERY PROCESSOR
# =============================================================================
class QueryMetrics:
    def __init__(self):
        self.classification_time = 0.0
        self.retrieval_time = 0.0
        self.response_time = 0.0
        self.total_time = 0.0
        self.query_type = ""

async def process_query(query: str, return_metrics: bool = False):
    metrics = QueryMetrics()
    total_start = time.time()
    query = query.strip()
    if not query:
        return ("Please ask a question.", metrics) if return_metrics else "Please ask a question."
    
    classify_start = time.time()
    query_type = await classify_query(query)
    metrics.classification_time = time.time() - classify_start
    metrics.query_type = query_type
    
    response = ""
    
    if query_type == "weather":
        response_start = time.time()
        response = await handle_weather_query(query)
        metrics.response_time = time.time() - response_start
    elif query_type == "document":
        retrieval_start = time.time()
        context, score = await rag_search(query)
        metrics.retrieval_time = time.time() - retrieval_start
        
        response_start = time.time()
        if context and score > 0.1:  # lower threshold to ensure answers
            response = await generate_rag_response(query, context)
        else:
            response = await generate_friendly_response(query)
        metrics.response_time = time.time() - response_start
    else:  # general
        response_start = time.time()
        response = await generate_friendly_response(query)
        metrics.response_time = time.time() - response_start
    
    metrics.total_time = time.time() - total_start
    if return_metrics:
        return response.strip(), metrics
    return response.strip()

def print_metrics(metrics: QueryMetrics):
    print("\n" + "="*50)
    print("INFERENCE METRICS")
    print("="*50)
    print(f"Query Type: {metrics.query_type}")
    print(f"Classification Time: {metrics.classification_time:.3f}s")
    print(f"Retrieval Time: {metrics.retrieval_time:.3f}s")
    print(f"Response Generation Time: {metrics.response_time:.3f}s")
    print(f"Total Time: {metrics.total_time:.3f}s")
    print("="*50 + "\n")

# =============================================================================
# CLI INTERFACE
# =============================================================================
async def main_cli():
    print("="*60)
    print("Async Smart AI Assistant")
    print("="*60)
    print("I can help you with:")
    print(" Weather information for any city")
    print(" Questions about uploaded documents")
    print(" General conversation")
    print("\nType 'exit' to quit, 'metrics on/off' to toggle metrics\n")
    
    show_metrics = True
    
    while True:
        q = input("You: ").strip()
        if q.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if q.lower() == "metrics on":
            show_metrics = True
            print("Metrics display: ON\n")
            continue
        if q.lower() == "metrics off":
            show_metrics = False
            print("Metrics display: OFF\n")
            continue
        
        response, metrics = await process_query(q, return_metrics=True)
        print(f"\nAssistant: {response}")
        if show_metrics:
            print_metrics(metrics)

if __name__ == "__main__":
    asyncio.run(main_cli())
