import streamlit as st
import tempfile
import os
import requests
import time
import numpy as np
import av
import logging
import warnings
from typing import Optional
from elevenlabs import ElevenLabs
from langchain_core.messages import HumanMessage
from src.beta_app import app as langgraph_app, AgentState, init_state
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from audio_recorder_streamlit import audio_recorder
from cryptography.utils import CryptographyDeprecationWarning

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DEBUG_MODE = True
MAX_RETRIES = 3
AUDIO_SAMPLE_RATE = 16000

# --- API Keys ---
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"

# --- Network Configuration ---
if os.name == 'nt':
    os.environ["GRPC_DNS_RESOLVER"] = "native"
    os.environ["AIOICE_NO_MDNS"] = "1"

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": "stun:global.stun.twilio.com:3478"}
    ],
    "iceTransportPolicy": "all"
}

# --- Initialize Clients ---
try:
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
except Exception as e:
    st.error(f"Failed to initialize ElevenLabs: {str(e)}")
    elevenlabs_client = None

# --- Audio Processor ---
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []
        self.sample_rate = AUDIO_SAMPLE_RATE
    
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame.to_ndarray())
        return frame

# --- Helper Functions ---
def save_audio_frames(frames: list, sample_rate: int = AUDIO_SAMPLE_RATE) -> Optional[str]:
    try:
        audio_data = np.concatenate(frames, axis=0)
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import scipy.io.wavfile
            scipy.io.wavfile.write(f.name, sample_rate, audio_data)
            return f.name
    except Exception as e:
        logger.error(f"Error saving audio: {str(e)}")
        return None

def transcribe_with_assemblyai(audio_path: str) -> tuple[Optional[str], Optional[str]]:
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    
    for attempt in range(MAX_RETRIES):
        try:
            # Upload audio
            with open(audio_path, "rb") as f:
                upload_resp = requests.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers=headers,
                    data=f,
                    timeout=30
                )
            if upload_resp.status_code != 200:
                return None, f"Upload failed (attempt {attempt + 1}): {upload_resp.text}"
            
            # Start transcription
            transcript_resp = requests.post(
                "https://api.assemblyai.com/v2/transcript",
                headers=headers,
                json={"audio_url": upload_resp.json()["upload_url"]},
                timeout=30
            )
            if transcript_resp.status_code != 200:
                return None, f"Transcription request failed (attempt {attempt + 1}): {transcript_resp.text}"
            
            # Poll for results
            transcript_id = transcript_resp.json()["id"]
            for _ in range(30):  # 60s total timeout
                poll_resp = requests.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                    headers=headers,
                    timeout=30
                )
                result = poll_resp.json()
                
                if result.get("status") == "completed":
                    return result["text"], None
                elif result.get("status") == "error":
                    break
                
                time.sleep(2)
            
            if attempt == MAX_RETRIES - 1:
                return None, "Transcription timed out after multiple attempts"
            time.sleep(1)  # Wait before retrying
            
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return None, f"Transcription failed after {MAX_RETRIES} attempts: {str(e)}"
            time.sleep(1)

def tts_with_elevenlabs(text: str) -> tuple[Optional[bytes], Optional[str]]:
    if not elevenlabs_client:
        return None, "ElevenLabs client not initialized"
    
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

# --- Session State ---
if "current_state" not in st.session_state:
    st.session_state.current_state = init_state.copy()
if "history" not in st.session_state:
    st.session_state.history = []
if "processing" not in st.session_state:
    st.session_state.processing = False

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
        
        if st.button("Check Network Status"):
            try:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.settimeout(3)
                    s.sendto(b"", ("stun.l.google.com", 19302))
                    st.success("STUN server reachable")
            except Exception as e:
                st.error(f"Network check failed: {str(e)}")

# --- Main Interface ---
st.title("AI Voice Interviewer (LangGraph)")

# --- Audio Capture ---
audio_source = st.radio(
    "Audio Input Method",
    ["WebRTC (Recommended)", "Basic Recorder"],
    horizontal=True,
    index=0
)

def process_audio_input(audio_input):
    if st.session_state.processing:
        return
    
    st.session_state.processing = True
    try:
        with st.spinner("Processing..."):
            # Save audio to temp file
            if isinstance(audio_input, list):  # WebRTC frames
                temp_audio_path = save_audio_frames(audio_input)
            else:  # Basic recorder bytes
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_input)
                    temp_audio_path = f.name
            
            if not temp_audio_path:
                raise Exception("Failed to save audio")
            
            # Transcribe
            user_text, stt_error = transcribe_with_assemblyai(temp_audio_path)
            if stt_error:
                raise Exception(stt_error)
            
            st.session_state.history.append(("You", user_text))
            st.audio(temp_audio_path)
            
            # Agent Processing
            st.session_state.current_state["messages"].append(
                HumanMessage(content=user_text)
            )
            
            result = langgraph_app.invoke(st.session_state.current_state)
            st.session_state.current_state = result
            
            ai_message = result["messages"][-1]
            ai_text = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
            st.session_state.history.append(("AI", ai_text))
            
            # TTS
            if elevenlabs_client:
                audio_bytes, tts_error = tts_with_elevenlabs(ai_text)
                if tts_error:
                    st.warning(f"TTS failed: {tts_error}")
                else:
                    st.audio(audio_bytes, format="audio/mp3")
            else:
                st.warning("ElevenLabs not configured - text only response")
            
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        logger.exception("Processing failed")
    finally:
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        st.session_state.processing = False
        st.rerun()

if audio_source == "WebRTC (Recommended)":
    audio_ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "audio": {
                "sampleRate": AUDIO_SAMPLE_RATE,
                "channelCount": 1,
                "echoCancellation": True,
                "noiseSuppression": True,
            },
            "video": False
        },
        audio_processor_factory=AudioRecorder,
        async_processing=True
    )
    
    if audio_ctx and audio_ctx.audio_processor and st.button("Submit WebRTC Recording"):
        process_audio_input(audio_ctx.audio_processor.frames)
else:
    audio_bytes = audio_recorder()
    if audio_bytes and st.button("Submit Basic Recording"):
        process_audio_input(audio_bytes)

# --- Conversation History ---
st.markdown("## Conversation History")
for sender, msg in st.session_state.history:
    st.markdown(f"**{sender}:** {msg}")

# --- Debug Information ---
if DEBUG_MODE:
    with st.expander("Debug Information"):
        st.write("Current State:", st.session_state.current_state)
        st.write("Environment Variables:", {
            "ASSEMBLYAI_API_KEY": "****" if ASSEMBLYAI_API_KEY else "Not set",
            "ELEVENLABS_API_KEY": "****" if ELEVENLABS_API_KEY else "Not set"
        })
        st.write("Processing Status:", st.session_state.processing)