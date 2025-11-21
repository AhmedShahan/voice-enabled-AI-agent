import streamlit as st
import io
import soundfile as sf
from rag2 import process_query

st.title("🎤 Voice Chatbot")

# Load ASR
@st.cache_resource
def load_asr():
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

asr = load_asr()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Voice input in sidebar
with st.sidebar:
    st.header("🎙️ Voice")
    audio = st.audio_input("Record")
    
    if audio:
        try:
            data, sr = sf.read(io.BytesIO(audio.getvalue()))
            sf.write("temp.wav", data, sr)
            text = asr("temp.wav")["text"]
            st.success(f"You said: {text}")
            st.session_state.voice_text = text
        except Exception as e:
            st.error(str(e))

# Text input
user_input = st.chat_input("Type here...")

# Get input
query = user_input
if not query and st.session_state.get("voice_text"):
    query = st.session_state.voice_text
    st.session_state.voice_text = None

# Process
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("..."):
            response = process_query(query)
            st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})