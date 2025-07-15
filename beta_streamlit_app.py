import streamlit as st
import tempfile
import os
import requests
import time
from elevenlabs import ElevenLabs
from langchain_core.messages import HumanMessage
from src.beta_app import app as langgraph_app, AgentState, init_state
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import numpy as np
from audio_recorder_streamlit import audio_recorder  # Fallback recorder
import logging

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DEBUG_MODE = True  # Toggle for debug outputs

# --- API Keys ---
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"

# --- Initialize Clients ---
try:
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize ElevenLabs: {str(e)}")
    st.stop()

# --- Helper Functions ---
def save_audio_frames(frames, sample_rate=16000):
    """Convert audio frames to WAV file."""
    audio_data = np.concatenate(frames, axis=0)
    if audio_data.dtype != np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        import scipy.io.wavfile
        scipy.io.wavfile.write(f.name, sample_rate, audio_data)
        return f.name

def transcribe_with_assemblyai(audio_path):
    """Robust transcription with error handling."""
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    
    try:
        # Upload
        with open(audio_path, "rb") as f:
            upload_resp = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers=headers,
                data=f,
                timeout=30
            )
        if upload_resp.status_code != 200:
            return None, f"Upload failed: {upload_resp.text}"
        
        # Transcribe
        transcript_resp = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            headers=headers,
            json={"audio_url": upload_resp.json()["upload_url"]},
            timeout=30
        )
        if transcript_resp.status_code != 200:
            return None, f"Transcription request failed: {transcript_resp.text}"
        
        # Poll
        transcript_id = transcript_resp.json()["id"]
        for _ in range(30):  # 30 attempts with 2s delay = 60s total
            poll_resp = requests.get(
                f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                headers=headers,
                timeout=30
            )
            result = poll_resp.json()
            
            if result.get("status") == "completed":
                return result["text"], None
            elif result.get("status") == "error":
                return None, f"Transcription error: {result.get('error')}"
            
            time.sleep(2)
        
        return None, "Transcription timed out"
    
    except Exception as e:
        return None, f"Transcription exception: {str(e)}"

def tts_with_elevenlabs(text):
    """Text-to-speech with error handling."""
    try:
        if not text or len(text.strip()) == 0:
            return None, "Empty text input"
            
        audio_stream = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL_ID,
            output_format="mp3_44100_128",
            timeout=30
        )
        audio_bytes = b"".join(list(audio_stream))
        return audio_bytes, None
    except Exception as e:
        return None, f"TTS Error: {str(e)}"

# --- Streamlit UI ---
st.set_page_config(page_title="AI Voice Interviewer", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("AI Voice Interviewer")
    st.markdown("""
    - Record your question and submit
    - AI will respond with voice and text
    - Powered by LangGraph, AssemblyAI, and ElevenLabs
    """)
    
    if DEBUG_MODE:
        st.markdown("---")
        st.subheader("Debug Panel")
        if st.button("Test Agent Directly"):
            test_state = init_state.copy()
            test_state["messages"].append(HumanMessage(content="Hello, can you hear me?"))
            try:
                result = langgraph_app.invoke(test_state)
                st.success(f"Agent response: {result['messages'][-1].content}")
            except Exception as e:
                st.error(f"Agent test failed: {str(e)}")

# --- Session State ---
if "current_state" not in st.session_state:
    st.session_state.current_state = init_state.copy()
if "history" not in st.session_state:
    st.session_state.history = []

# --- Main Interface ---
st.title("AI Voice Interviewer (LangGraph)")
st.write("Record your question and submit. The AI will respond with voice and text.")

# --- Audio Capture ---
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []
    
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame.to_ndarray())
        return frame

# WebRTC Streamer with fallback
audio_ctx = None
try:
    audio_ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:stun1.l.google.com:19302"}
            ]
        },
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=AudioRecorder,
        async_processing=True
    )
except Exception as e:
    st.warning(f"WebRTC initialization warning: {str(e)}")
    st.info("Falling back to basic audio recorder")
    audio_bytes = audio_recorder()
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_audio_path = f.name

# --- Processing Logic ---
if (audio_ctx and audio_ctx.audio_processor and audio_ctx.audio_processor.frames) or ("temp_audio_path" in locals()):
    if st.button("Submit Recording"):
        with st.spinner("Processing..."):
            try:
                # Get audio file
                if audio_ctx and audio_ctx.audio_processor:
                    temp_audio_path = save_audio_frames(audio_ctx.audio_processor.frames)
                
                # Transcribe
                user_text, stt_error = transcribe_with_assemblyai(temp_audio_path)
                if DEBUG_MODE:
                    st.write(f"DEBUG - Transcription: {user_text}")
                
                if stt_error:
                    raise Exception(f"STT Error: {stt_error}")
                
                st.session_state.history.append(("You", user_text))
                st.audio(temp_audio_path)
                
                # Agent Processing
                st.session_state.current_state["messages"].append(
                    HumanMessage(content=user_text)
                )
                
                if DEBUG_MODE:
                    st.write(f"DEBUG - Current state: {st.session_state.current_state}")
                
                result = langgraph_app.invoke(st.session_state.current_state)
                st.session_state.current_state = result
                
                ai_message = result["messages"][-1]
                ai_text = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
                st.session_state.history.append(("AI", ai_text))
                
                if DEBUG_MODE:
                    st.write(f"DEBUG - Agent raw output: {result}")
                
                # TTS
                audio_bytes, tts_error = tts_with_elevenlabs(ai_text)
                if tts_error:
                    raise Exception(f"TTS Error: {tts_error}")
                
                st.audio(audio_bytes, format="audio/mp3")
                
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                logger.exception("Processing failed")
            finally:
                if "temp_audio_path" in locals() and os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

# --- Conversation History ---
st.markdown("## Conversation History")
for sender, msg in st.session_state.history:
    st.markdown(f"**{sender}:** {msg}")

# --- Debug Outputs ---
if DEBUG_MODE:
    with st.expander("Debug Information"):
        st.write("Current session state:", st.session_state.current_state)
        st.write("Environment variables:", {
            "ASSEMBLYAI_API_KEY": "****" if ASSEMBLYAI_API_KEY else "Not set",
            "ELEVENLABS_API_KEY": "****" if ELEVENLABS_API_KEY else "Not set"
        })