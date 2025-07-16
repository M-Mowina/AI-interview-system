import streamlit as st
import os
import requests
import time
import tempfile
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from elevenlabs import ElevenLabs

# Set page config
st.set_page_config(page_title="Voice Assistant", layout="wide")

# --- Configuration ---
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Default voice (Bella)
ELEVENLABS_MODEL = "eleven_monolingual_v2"

# --- Audio Recording Function ---
def record_audio(duration=5, sample_rate=16000):
    """Record audio using sounddevice"""
    st.info(f"🎙 Recording for {duration} seconds... (Speak now)")
    audio = sd.rec(int(duration * sample_rate), 
                  samplerate=sample_rate,
                  channels=1, 
                  dtype='int16')
    sd.wait()
    return audio, sample_rate

# --- Transcription Function (Your Working Version) ---
def assemblyai_transcribe(audio, sample_rate):
    """Transcribe audio using AssemblyAI"""
    # Save to temp wav file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        write_wav(tmp.name, sample_rate, audio)
        audio_path = tmp.name

    # Upload to AssemblyAI
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    with open(audio_path, "rb") as f:
        upload_resp = requests.post("https://api.assemblyai.com/v2/upload", 
                                  headers=headers, 
                                  data=f)

    if upload_resp.status_code != 200:
        st.error(f"Upload failed: {upload_resp.text}")
        os.remove(audio_path)
        return None

    upload_url = upload_resp.json()["upload_url"]

    # Request transcription
    transcript_resp = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=headers,
        json={"audio_url": upload_url}
    )

    if transcript_resp.status_code != 200:
        st.error(f"Transcription request failed: {transcript_resp.text}")
        os.remove(audio_path)
        return None

    transcript_id = transcript_resp.json()["id"]

    # Poll until complete
    with st.spinner("Transcribing..."):
        while True:
            time.sleep(1.5)
            poll_resp = requests.get(
                f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                headers=headers
            )
            result = poll_resp.json()
            status = result.get("status")
            
            if status == "completed":
                os.remove(audio_path)
                return result["text"]
            elif status == "error":
                st.error(f"Transcription error: {result.get('error')}")
                os.remove(audio_path)
                return None

# --- Text-to-Speech Function ---
def text_to_speech(text):
    """Convert text to speech using ElevenLabs"""
    if not text:
        return None, "Empty text input"
    
    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        return b"".join(list(audio_stream)), None
    except Exception as e:
        return None, str(e)

# --- App UI ---
st.title("🎙️ Voice Assistant")
st.markdown("""
Press record, speak for up to 5 seconds, and hear the AI response!
""")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Audio recording
if st.button("🎤 Record Voice (5 seconds)"):
    if not ASSEMBLYAI_API_KEY:
        st.error("AssemblyAI API key not found!")
    else:
        # Record audio
        audio, sample_rate = record_audio()
        
        # Play back recording
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            write_wav(tmp.name, sample_rate, audio)
            st.audio(tmp.name, format="audio/wav")
            audio_path = tmp.name
        
        # Transcribe
        transcribed_text = assemblyai_transcribe(audio, sample_rate)
        
        if transcribed_text:
            st.session_state.conversation.append(("You", transcribed_text))
            
            # Display transcription
            st.subheader("Transcription")
            st.write(transcribed_text)
            
            # Generate AI response (simple echo for demo)
            ai_response = f"I heard you say: {transcribed_text}"
            st.session_state.conversation.append(("AI", ai_response))
            
            # Convert to speech
            if ELEVENLABS_API_KEY:
                audio_response, tts_error = text_to_speech(ai_response)
                
                if tts_error:
                    st.error(f"Voice generation failed: {tts_error}")
                elif audio_response:
                    st.subheader("AI Voice Response")
                    st.audio(audio_response, format="audio/mp3")
            else:
                st.warning("ElevenLabs API key not configured - text only")

# Display conversation history
if st.session_state.conversation:
    st.subheader("Conversation History")
    for speaker, text in st.session_state.conversation:
        st.markdown(f"**{speaker}:** {text}")

# --- Deployment Help ---
st.markdown("---")
st.markdown("""
### How to Deploy:
1. Add these secrets to your Streamlit app:
```toml
# .streamlit/secrets.toml
ASSEMBLYAI_API_KEY = "your_assemblyai_key"
ELEVENLABS_API_KEY = "your_elevenlabs_key""")