import streamlit as st
import io
import soundfile as sf
import asyncio
from rag2 import process_query

# Page config
st.set_page_config(page_title="Voice Chatbot", page_icon="🎤")
st.title("🎤 Voice Chatbot with RAG")

# Load ASR model (cached)
@st.cache_resource
def load_asr_model():
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

asr = load_asr_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper: run async function
def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Sidebar for voice input
with st.sidebar:
    st.header("🎙️ Voice Input")
    audio = st.audio_input("Record your question")
    
    if audio:
        with st.spinner("Transcribing..."):
            try:
                audio_bytes = audio.getvalue()
                audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                sf.write("temp.wav", audio_data, sample_rate)
                result = asr("temp.wav")
                transcribed = result["text"]
                st.success(f"You said: {transcribed}")
                
                # Store transcription to process
                st.session_state.voice_text = transcribed
            except Exception as e:
                st.error(f"Transcription failed: {e}")

# Text input (chat style)
user_input = st.chat_input("Type your message here...")

# Check if we have input from voice or text
query = None
if user_input:
    query = user_input
elif "voice_text" in st.session_state and st.session_state.voice_text:
    query = st.session_state.voice_text
    st.session_state.voice_text = None  # Clear after use

# Process the query
if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_async(process_query(query))
            st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()