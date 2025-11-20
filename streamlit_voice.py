import streamlit as st
import io
import soundfile as sf

st.title("Simple Voice Recorder")

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

    # Provide download button
    with open(output_audio_file, "rb") as f:
        st.download_button(
            label="Download WAV file",
            data=f,
            file_name=output_audio_file,
            mime="audio/wav",
        )
