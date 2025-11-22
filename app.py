import streamlit as st
import io
import time
import soundfile as sf
from dataclasses import dataclass
from typing import Optional
from src.agentrag import process_query, QueryMetrics

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="🎤 Voice AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# =============================================================================
# METRICS DATACLASS
# =============================================================================
@dataclass
class FullMetrics:
    """Complete metrics including STT."""
    stt_time: float = 0.0
    classification_time: float = 0.0
    retrieval_time: float = 0.0
    response_time: float = 0.0
    total_time: float = 0.0
    query_type: str = ""

# =============================================================================
# LOAD ASR MODEL (CACHED)
# =============================================================================
@st.cache_resource
def load_asr_model():
    """Load Whisper ASR model (cached for performance)."""
    from transformers import pipeline
    return pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-tiny",
        chunk_length_s=30
    )

# =============================================================================
# TRANSCRIBE AUDIO
# =============================================================================
def transcribe_audio(audio_data, asr_model) -> tuple[str, float]:
    """
    Transcribe audio to text using Whisper.
    Returns: (transcribed_text, stt_time)
    """
    start_time = time.time()
    
    try:
        # Read audio data
        data, sample_rate = sf.read(io.BytesIO(audio_data))
        
        # Save temporary file for ASR
        temp_path = "temp_audio.wav"
        sf.write(temp_path, data, sample_rate)
        
        # Transcribe
        result = asr_model(temp_path)
        text = result["text"].strip()
        
        stt_time = time.time() - start_time
        return text, stt_time
        
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# =============================================================================
# DISPLAY METRICS
# =============================================================================
def display_metrics(metrics: FullMetrics):
    """Display metrics in a nice format."""
    st.markdown("---")
    st.markdown("#### 📊 Inference Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Query Type", metrics.query_type.upper())
        st.metric("STT Time", f"{metrics.stt_time:.3f}s")
    
    with col2:
        st.metric("Classification", f"{metrics.classification_time:.3f}s")
        st.metric("Retrieval", f"{metrics.retrieval_time:.3f}s")
    
    with col3:
        st.metric("Response Gen", f"{metrics.response_time:.3f}s")
        st.metric("Total Time", f"{metrics.total_time:.3f}s")

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Header
    st.title("🎤 Voice AI Assistant")
    st.markdown("""
    **I can help you with:**
    - 🌤️ **Weather** - Ask about weather in any city
    - 📚 **Documents** - Query your uploaded documents
    - 💬 **Chat** - General conversation
    """)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_metrics" not in st.session_state:
        st.session_state.show_metrics = True
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = None
    
    # Load ASR model
    with st.spinner("Loading speech recognition model..."):
        asr_model = load_asr_model()
    
    # ==========================================================================
    # SIDEBAR - Voice Input & Settings
    # ==========================================================================
    with st.sidebar:
        st.header("🎙️ Voice Input")
        
        # Voice recording
        audio = st.audio_input("Click to record your question")
        
        # Settings
        st.markdown("---")
        st.header("⚙️ Settings")
        st.session_state.show_metrics = st.checkbox(
            "Show Inference Metrics", 
            value=st.session_state.show_metrics
        )
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.last_metrics = None
            st.rerun()
        
        # Info
        st.markdown("---")
        st.markdown("""
        **💡 Tips:**
        - Say "weather in [city]" for weather
        - Ask questions about your documents
        - Metrics show processing times
        """)
    
    # ==========================================================================
    # PROCESS VOICE INPUT
    # ==========================================================================
    voice_text = None
    stt_time = 0.0
    
    if audio:
        with st.sidebar:
            with st.spinner("🎧 Transcribing..."):
                voice_text, stt_time = transcribe_audio(audio.getvalue(), asr_model)
                
                if voice_text and not voice_text.startswith("Error"):
                    st.success(f"**You said:** {voice_text}")
                else:
                    st.error(voice_text)
                    voice_text = None
    
    # ==========================================================================
    # CHAT DISPLAY
    # ==========================================================================
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    
    # ==========================================================================
    # TEXT INPUT
    # ==========================================================================
    user_input = st.chat_input("Type your question here...")
    
    # Determine final query (voice takes priority if just recorded)
    query = None
    current_stt_time = 0.0
    
    if voice_text:
        query = voice_text
        current_stt_time = stt_time
    elif user_input:
        query = user_input
        current_stt_time = 0.0  # No STT for typed input
    
    # ==========================================================================
    # PROCESS QUERY
    # ==========================================================================
    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.write(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                total_start = time.time()
                
                # Get response with metrics
                response, rag_metrics = process_query(query, return_metrics=True)
                
                total_time = time.time() - total_start
                
                # Display response
                st.write(response)
                
                # Store full metrics
                full_metrics = FullMetrics(
                    stt_time=current_stt_time,
                    classification_time=rag_metrics.classification_time,
                    retrieval_time=rag_metrics.retrieval_time,
                    response_time=rag_metrics.response_time,
                    total_time=total_time + current_stt_time,
                    query_type=rag_metrics.query_type
                )
                st.session_state.last_metrics = full_metrics
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display metrics if enabled
        if st.session_state.show_metrics and st.session_state.last_metrics:
            display_metrics(st.session_state.last_metrics)
    
    # ==========================================================================
    # SHOW LAST METRICS (if no new query but metrics exist)
    # ==========================================================================
    elif st.session_state.show_metrics and st.session_state.last_metrics and st.session_state.messages:
        display_metrics(st.session_state.last_metrics)

# =============================================================================
# RUN APP
# =============================================================================
if __name__ == "__main__":
    main()