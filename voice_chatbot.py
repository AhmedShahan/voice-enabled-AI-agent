import streamlit as st
import io
import soundfile as sf
from transformers import pipeline

st.set_page_config(page_title="Voice Chatbot", layout="wide")

# Load ASR model (cached)
@st.cache_resource
def load_asr_model():
    return pipeline("automatic-speech-recognition")

asr = load_asr_model()

st.title("Simple Voice Chatbot")

# Sidebar input
st.sidebar.header("Input Options")

# Voice input
voice_input = st.sidebar.audio_input("Record your voice")

user_text = ""

if voice_input:
    with io.BytesIO(voice_input.getbuffer()) as audio_buffer:
        try:
            audio_data, sample_rate = sf.read(audio_buffer)
            sf.write("temp_audio.wav", audio_data, sample_rate)
            with st.spinner("Transcribing voice..."):
                result = asr("temp_audio.wav", chunk_length_s=30)
                user_text = result["text"]
            st.sidebar.success("Transcription done")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# Text input
text_input = st.sidebar.text_area("Or type your message:")
if text_input:
    user_text = text_input

# Display chat
if user_text:
    st.subheader("User:")
    st.write(user_text)

    # Simple response (echo)
    st.subheader("Bot:")
    bot_response = f"You said: {user_text}"
    st.write(bot_response)
