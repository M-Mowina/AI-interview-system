import streamlit as st
import os
import requests
import time
import tempfile
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from elevenlabs import ElevenLabs
from audio_recorder_streamlit import audio_recorder  # NEW: import the widget

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

# --- Transcription Function (Refactored with Timeout) ---
def assemblyai_transcribe(audio_path, max_wait=60):
    """Transcribe audio using AssemblyAI from a WAV file path, with timeout and progress."""
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    with st.status("Uploading audio to AssemblyAI...", expanded=True) as status:
        with open(audio_path, "rb") as f:
            upload_resp = requests.post(
                "https://api.assemblyai.com/v2/upload", headers=headers, data=f
            )
        if upload_resp.status_code != 200:
            status.update(label="Upload failed", state="error")
            st.error(f"Upload failed: {upload_resp.text}")
            os.remove(audio_path)
            return None
        status.update(label="Audio uploaded. Requesting transcription...", state="running")

        upload_url = upload_resp.json()["upload_url"]
        transcript_resp = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            headers=headers,
            json={"audio_url": upload_url}
        )
        if transcript_resp.status_code != 200:
            status.update(label="Transcription request failed", state="error")
            st.error(f"Transcription request failed: {transcript_resp.text}")
            os.remove(audio_path)
            return None

        transcript_id = transcript_resp.json()["id"]
        status.update(label="Transcription requested. Waiting for result...", state="running")

        # Poll until complete or timeout
        start_time = time.time()
        while True:
            if time.time() - start_time > max_wait:
                status.update(label="Transcription timed out", state="error")
                st.error("Transcription timed out. Please try again or record a shorter clip.")
                os.remove(audio_path)
                return None
            time.sleep(1.5)
            poll_resp = requests.get(
                f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                headers=headers
            )
            result = poll_resp.json()
            status_str = result.get("status")
            if status_str == "completed":
                status.update(label="Transcription complete!", state="complete")
                os.remove(audio_path)
                return result["text"]
            elif status_str == "error":
                status.update(label="Transcription error", state="error")
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
Press record, speak, and hear the AI response!
""")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# --- Flexible Audio Recording (NEW) ---
MAX_RECORD_SECONDS = 30
st.subheader("Flexible Voice Recording")
st.info(f"Please keep your recording under {MAX_RECORD_SECONDS} seconds for best results.")
audio_bytes = audio_recorder()

if audio_bytes:
    import wave
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name
    # Check duration
    with wave.open(audio_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    if duration > MAX_RECORD_SECONDS:
        st.error(f"Recording is {duration:.1f} seconds. Please record less than {MAX_RECORD_SECONDS} seconds.")
        os.remove(audio_path)
    else:
        st.audio(audio_bytes, format="audio/wav")
        transcribed_text = assemblyai_transcribe(audio_path, max_wait=60)
        if transcribed_text:
            st.session_state.conversation.append(("You", transcribed_text))
            st.subheader("Transcription")
            st.write(transcribed_text)
            ai_response = f"I heard you say: {transcribed_text}"
            st.session_state.conversation.append(("AI", ai_response))
            if ELEVENLABS_API_KEY:
                audio_response, tts_error = text_to_speech(ai_response)
                if tts_error:
                    st.error(f"Voice generation failed: {tts_error}")
                elif audio_response:
                    st.subheader("AI Voice Response")
                    st.audio(audio_response, format="audio/mp3")
            else:
                st.warning("ElevenLabs API key not configured - text only")

# --- Fallback: Old 5-second Button (commented out) ---
# if st.button("🎤 Record Voice (5 seconds)"):
#     if not ASSEMBLYAI_API_KEY:
#         st.error("AssemblyAI API key not found!")
#     else:
#         audio, sample_rate = record_audio()
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#             write_wav(tmp.name, sample_rate, audio)
#             st.audio(tmp.name, format="audio/wav")
#             audio_path = tmp.name
#         # Check duration
#         import wave
#         with wave.open(audio_path, "rb") as wf:
#             frames = wf.getnframes()
#             rate = wf.getframerate()
#             duration = frames / float(rate)
#         if duration > MAX_RECORD_SECONDS:
#             st.error(f"Recording is {duration:.1f} seconds. Please record less than {MAX_RECORD_SECONDS} seconds.")
#             os.remove(audio_path)
#         else:
#             transcribed_text = assemblyai_transcribe(audio_path, max_wait=60)
#             if transcribed_text:
#                 st.session_state.conversation.append(("You", transcribed_text))
#                 st.subheader("Transcription")
#                 st.write(transcribed_text)
#                 ai_response = f"I heard you say: {transcribed_text}"
#                 st.session_state.conversation.append(("AI", ai_response))
#                 if ELEVENLABS_API_KEY:
#                     audio_response, tts_error = text_to_speech(ai_response)
#                     if tts_error:
#                         st.error(f"Voice generation failed: {tts_error}")
#                     elif audio_response:
#                         st.subheader("AI Voice Response")
#                         st.audio(audio_response, format="audio/mp3")
#                 else:
#                     st.warning("ElevenLabs API key not configured - text only")

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