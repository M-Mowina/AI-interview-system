import os
import streamlit as st
import wave
import tempfile
from audio_recorder_streamlit import audio_recorder
from utils.audio_utils import whisper_transcribe, elevenlabs_tts
from src.beta_app import app as langgraph_app, init_state
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="AI Voice Interviewer (Whisper)", layout="wide")

# --- Session State ---
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = init_state.copy()

# --- Sidebar ---
with st.sidebar:
    st.title("AI Voice Interviewer")
    st.markdown("""
    - Record your question and submit
    - AI will respond with voice and text
    - Powered by LangGraph, Whisper, and ElevenLabs
    """)
    
    # Whisper model selection
    st.subheader("Whisper Settings")
    model_size = st.selectbox(
        "Whisper Model Size",
        ["tiny", "base", "small", "medium", "large"],
        index=1,  # Default to "base"
        help="Larger models are more accurate but slower"
    )
    
    st.info(f"""
    **Model Sizes:**
    - tiny: ~39MB, fastest
    - base: ~74MB, good balance (default)
    - small: ~244MB, better accuracy
    - medium: ~769MB, high accuracy
    - large: ~1550MB, best accuracy
    """)

# --- Main Interface ---
st.title("AI Voice Interviewer (LangGraph + Whisper)")
MAX_RECORD_SECONDS = 30
st.subheader("Flexible Voice Conversation with AI Agent")
st.info(f"Please keep your recording under {MAX_RECORD_SECONDS} seconds for best results.")

# Simple audio recorder with better UI
st.info("💡 **Tip**: Click the microphone button to start recording, then click it again to stop. The recording will process automatically.")
audio_bytes = audio_recorder(
    text="🎤 Click to start/stop recording",
    recording_color="#e74c3c",
    neutral_color="#95a5a6",
    icon_name="microphone",
    icon_size="2x"
)

if audio_bytes:
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
        with st.status(f"Transcribing audio with Whisper ({model_size} model)...", expanded=True) as status:
            transcribed_text, stt_error = whisper_transcribe(audio_path, model_size=model_size)
            if stt_error:
                status.update(label="Transcription failed", state="error")
                st.error(stt_error)
            elif transcribed_text:
                status.update(label="Transcription complete!", state="complete")
                st.session_state.conversation.append(("You", transcribed_text))
                st.subheader("Transcription")
                st.write(transcribed_text)
                # --- LangGraph agent workflow ---
                st.session_state.agent_state["messages"].append(HumanMessage(content=transcribed_text))
                result = langgraph_app.invoke(st.session_state.agent_state)
                st.session_state.agent_state = result
                ai_message = result["messages"][-1]
                ai_text = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
                st.session_state.conversation.append(("AI", ai_text))
                st.subheader("AI Response")
                st.write(ai_text)
                audio_path, tts_error = elevenlabs_tts(ai_text)
                if tts_error:
                    st.error(f"Voice generation failed: {tts_error}")
                elif audio_path:
                    st.subheader("AI Voice Response")
                    st.audio(audio_path, format="audio/mp3")

# --- Conversation History ---
if st.session_state.conversation:
    st.subheader("Conversation History")
    for speaker, text in st.session_state.conversation:
        st.markdown(f"**{speaker}:** {text}")

# --- Footer ---
st.markdown("---")
st.markdown("""
**Powered by:**
- **Whisper** (OpenAI) - Speech-to-Text
- **ElevenLabs** - Text-to-Speech  
- **LangGraph** - AI Agent
- **Streamlit** - Web Interface
""") 