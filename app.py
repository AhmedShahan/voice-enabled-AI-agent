"""
Streamlit Voice AI Assistant
Clean, functional UI for RAG with voice input via Whisper
"""

import streamlit as st
import tempfile
import time
import asyncio
from pathlib import Path

# Import your RAG processor
from src.agentrag import process_query, QueryMetrics

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Voice AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# =============================================================================
# LOAD WHISPER MODEL (CACHED)
# =============================================================================
@st.cache_resource
def load_whisper():
    """Load Whisper model once and cache it."""
    from transformers import pipeline
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        chunk_length_s=30
    )

# =============================================================================
# TRANSCRIBE AUDIO
# =============================================================================
def transcribe_audio(audio_bytes) -> tuple[str, float]:
    """Transcribe audio bytes to text."""
    start = time.time()
    
    try:
        asr = load_whisper()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        result = asr(temp_path)
        text = result["text"].strip()
        
        Path(temp_path).unlink(missing_ok=True)
        
        return text, time.time() - start
    
    except Exception as e:
        return f"Transcription error: {e}", 0.0

# =============================================================================
# ASYNC HELPER
# =============================================================================
def run_async(coro):
    """Run async function in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_metrics" not in st.session_state:
    st.session_state.show_metrics = True
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

# =============================================================================
# HEADER
# =============================================================================
st.title("🤖 Voice AI Assistant")
st.caption("Chat or speak - I can help with weather, documents, and more!")

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("🎙️ Voice Input")
    audio_input = st.audio_input("Record your question")
    
    st.divider()
    
    st.header("⚙️ Settings")
    st.session_state.show_metrics = st.checkbox(
        "Show metrics", 
        value=st.session_state.show_metrics
    )
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_audio_id = None
        st.rerun()
    
    st.divider()
    
    st.markdown("""
    **What I can do:**
    - 🌤️ Weather for any city
    - 📚 Answer from documents
    - 💬 General chat
    """)

# =============================================================================
# DISPLAY CHAT HISTORY
# =============================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("metrics") and st.session_state.show_metrics:
            with st.expander("📊 Metrics"):
                m = msg["metrics"]
                cols = st.columns(4)
                cols[0].metric("Type", m.get("type", "N/A"))
                cols[1].metric("STT", f"{m.get('stt', 0):.2f}s")
                cols[2].metric("Process", f"{m.get('process', 0):.2f}s")
                cols[3].metric("Total", f"{m.get('total', 0):.2f}s")

# =============================================================================
# PROCESS INPUT
# =============================================================================
query = None
stt_time = 0.0

# Handle voice input - CHECK IF IT'S NEW AUDIO
if audio_input:
    # Create unique ID for this audio based on its content
    audio_id = hash(audio_input.getvalue())
    
    # Only process if this is NEW audio (not already processed)
    if audio_id != st.session_state.last_audio_id:
        with st.spinner("🎧 Transcribing..."):
            query, stt_time = transcribe_audio(audio_input.getvalue())
            if query and not query.startswith("Transcription error"):
                st.sidebar.success(f"**Heard:** {query}")
                # Mark this audio as processed
                st.session_state.last_audio_id = audio_id
            else:
                st.sidebar.error(query)
                query = None

# Handle text input
text_input = st.chat_input("Type your message...")
if text_input:
    query = text_input
    stt_time = 0.0

# =============================================================================
# GENERATE RESPONSE
# =============================================================================
if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.write(query)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            start = time.time()
            
            response, metrics = run_async(process_query(query, return_metrics=True))
            
            process_time = time.time() - start
            total_time = process_time + stt_time
        
        st.write(response)
        
        # Store metrics
        msg_metrics = {
            "type": metrics.query_type,
            "stt": stt_time,
            "process": process_time,
            "total": total_time,
            "classification": metrics.classification_time,
            "retrieval": metrics.retrieval_time,
            "response": metrics.response_time
        }
        
        # Show metrics inline
        if st.session_state.show_metrics:
            with st.expander("📊 Metrics", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Type", metrics.query_type.upper())
                c2.metric("STT", f"{stt_time:.2f}s")
                c3.metric("Process", f"{process_time:.2f}s")
                c4.metric("Total", f"{total_time:.2f}s")
    
    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "metrics": msg_metrics
    })
    
    st.rerun()