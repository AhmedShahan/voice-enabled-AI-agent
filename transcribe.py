# transcriber.py

import torch
from transformers import pipeline
import streamlit as st

# --- Configuration ---
HUGGINGFACE_MODEL = "openai/whisper-tiny" 

@st.cache_resource
def load_whisper_pipeline(model_name: str):
    """Loads the Hugging Face Whisper pipeline only once."""
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
             device = "mps"
        
        st.caption(f"Backend: Loading Whisper model '{model_name}' on {device.upper()}...")
        
        # We pass the chunking parameters here to enable long-form transcription.
        # The 'stride' warning can sometimes be ignored as chunking is the key.
        pipe = pipeline(
            "automatic-speech-recognition", 
            model=model_name, 
            device=device,
            chunk_length_s=15, 
            stride=(3, 3)     
        )
        st.caption("Backend: Whisper model loaded successfully.")
        return pipe
        
    except Exception as e:
        st.error(f"Backend Error: Failed to load Hugging Face model. {e}")
        return None

# Load the pipeline globally
WHISPER_PIPELINE = load_whisper_pipeline(HUGGINGFACE_MODEL)

def transcribe_audio_file(audio_path: str) -> str:
    """Transcribes a local audio file using the pre-loaded pipeline."""
    if WHISPER_PIPELINE is None:
        return "ERROR: Transcription model not loaded. Check installation/logs."
        
    try:
        result = WHISPER_PIPELINE(audio_path)
        return result["text"]

    except Exception as e:
        return f"ERROR during transcription: {e}"