import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai import configure, GenerativeModel
import os
from transformers import pipeline

# ----------------------------
# Whisper ASR
# ----------------------------
def load_asr():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

asr = load_asr()

# ----------------------------
# API Key
# ----------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# LangChain Gemini (LLM only)
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.8
)

# Official Gemini SDK for TTS
configure(api_key=api_key)
tts_model = GenerativeModel("gemini-2.5-flash")

st.title("Real-Time Voice Chat (Whisper + Gemini)")
st.write("Speak and hear the response.")

audio_input = st.audio_input("Speak now")

if audio_input:
    audio_bytes = audio_input.getvalue()

    # ----------------------------
    # STT using Whisper Tiny
    # ----------------------------
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    asr_result = asr("temp.wav")
    user_text = asr_result["text"]
    st.write("You said:", user_text)

    # ----------------------------
    # LLM using LangChain wrapper
    # ----------------------------
    reply = chat_model.invoke(user_text)
    answer_text = reply.content
    st.write("Gemini:", answer_text)

    # ----------------------------
    # TTS using Google SDK (NOT LangChain)
    # ----------------------------
    tts_audio = tts_model.generate_content(
        answer_text,
        generation_config={
            "response_mime_type": "audio/wav"
        }
    )

    audio_out = tts_audio.generated_content[0].data
    st.audio(audio_out, format="audio/wav")
