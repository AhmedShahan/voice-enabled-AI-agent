import streamlit as st
import io
import soundfile as sf
from transformers import pipeline

st.title("Simple Voice Recorder with Transcription")

# Initialize the ASR pipeline (cached to avoid reloading)
@st.cache_resource
def load_asr_model():
    return pipeline("automatic-speech-recognition")

asr = load_asr_model()

# Record audio using Streamlit's built-in audio widget
audio = st.audio_input("Record audio (click to start/stop)")

if audio:
    # Play the recorded audio
    st.audio(audio, format="audio/wav")
    
    # Save audio as WAV
    output_audio_file = "recorded_audio.wav"
    try:
        with io.BytesIO(audio.getbuffer()) as audio_buffer:
            audio_data, sample_rate = sf.read(audio_buffer)
            sf.write(output_audio_file, audio_data, sample_rate, format="WAV")
        st.success(f"Audio saved as '{output_audio_file}'")
    except Exception as e:
        st.error(f"Error saving audio: {e}")
    
    # Transcribe audio
    st.subheader("Transcription")
    with st.spinner("Transcribing audio..."):
        try:
            # Use chunk_length_s for long audio to avoid memory issues
            result = asr(output_audio_file, chunk_length_s=30, return_timestamps="word")
            transcribed_text = result["text"]
            
            # Display transcription in a nice text area
            st.text_area("Transcribed Text:", transcribed_text, height=150)
            
            # Option to copy transcription
            st.code(transcribed_text, language=None)
            
            # Optionally show chunks with timestamps if available
            if "chunks" in result and result["chunks"]:
                with st.expander("View Word-level Timestamps"):
                    for chunk in result["chunks"]:
                        timestamp = chunk.get("timestamp", (0, 0))
                        text = chunk.get("text", "")
                        st.write(f"**[{timestamp[0]:.2f}s - {timestamp[1]:.2f}s]** {text}")
            
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
    
    # Provide download button for audio
    with open(output_audio_file, "rb") as f:
        st.download_button(
            label="Download WAV file",
            data=f,
            file_name=output_audio_file,
            mime="audio/wav",
        )