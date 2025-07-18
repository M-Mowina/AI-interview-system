import os
import requests
import time
import tempfile
import assemblyai as aai
from elevenlabs.client import ElevenLabs
import httpx
import whisper

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# --- Transcription Function (AssemblyAI, no Streamlit UI) ---
def assemblyai_transcribe(audio_path, max_wait=60):
    """Transcribe audio using AssemblyAI from a WAV file path, with timeout. Returns (text, error)"""
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    with open(audio_path, "rb") as f:
        upload_resp = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=f)
    if upload_resp.status_code != 200:
        os.remove(audio_path)
        return None, f"Upload failed: {upload_resp.text}"
    upload_url = upload_resp.json()["upload_url"]
    transcript_resp = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=headers,
        json={"audio_url": upload_url}
    )
    if transcript_resp.status_code != 200:
        os.remove(audio_path)
        return None, f"Transcription request failed: {transcript_resp.text}"
    transcript_id = transcript_resp.json()["id"]
    start_time = time.time()
    while True:
        if time.time() - start_time > max_wait:
            os.remove(audio_path)
            return None, "Transcription timed out. Please try again or record a shorter clip."
        time.sleep(1.5)
        poll_resp = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
            headers=headers
        )
        result = poll_resp.json()
        status = result.get("status")
        if status == "completed":
            os.remove(audio_path)
            return result["text"], None
        elif status == "error":
            os.remove(audio_path)
            return None, f"Transcription error: {result.get('error')}"

def transcribe_audio_file(audio_file_path):
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    transcriber = aai.Transcriber()
    try:
        transcript = transcriber.transcribe(audio_file_path)
    except httpx.ReadTimeout:
        return None, "Transcription timed out (network/server slow). Try a shorter recording or check your connection."
    except Exception as e:
        return None, f"Transcription failed: {e}"
    if transcript.status == aai.TranscriptStatus.error:
        return None, f"Transcription failed: {transcript.error}"
    return transcript.text, None

# Whisper STT (local transcription, no API key needed)
def whisper_transcribe(audio_file_path, model_size="base", language="en"):
    """
    Transcribe audio using OpenAI's Whisper model locally.
    
    Args:
        audio_file_path (str): Path to the audio file
        model_size (str): Whisper model size - "tiny", "base", "small", "medium", "large"
        language (str): Language code (e.g., "en" for English, "auto" for auto-detection)
    
    Returns:
        tuple: (transcribed_text, error_message) or (None, error_message) if failed
    """
    try:
        # Load the Whisper model (will download on first use)
        model = whisper.load_model(model_size)
        
        # Transcribe the audio file with language specification
        # When language is specified, Whisper will force the recognition in that language
        result = model.transcribe(
            audio_file_path,
            language=language if language != "auto" else None,  # None means auto-detect
            fp16=False  # Use fp16=False for better compatibility on some systems
        )
        
        # Return the transcribed text
        return result["text"], None
        
    except Exception as e:
        return None, f"Whisper transcription failed: {str(e)}"

# ElevenLabs TTS (for version 2.7.1 with client API, returns path to temp mp3 file or error)
def elevenlabs_tts(text, api_key=None, voice_id="JBFqnCBsd6RMkjVDRZzb"):
    """
    Generates speech from text using ElevenLabs API (version 2.7.1) and saves it to a temporary MP3 file.

    Args:
        text (str): The text to convert to speech.
        api_key (str, optional): Your ElevenLabs API key. If not provided, it will attempt
                                 to use the ELEVENLABS_API_KEY environment variable.
        voice_id (str): The ID of the ElevenLabs voice to use.
                        Default is "JBFqnCBsd6RMkjVDRZzb".

    Returns:
        tuple: (audio_path, error_message) or (None, error_message) if failed
    """
    if not text:
        return None, "Empty text input"
    
    try:
        # Initialize the ElevenLabs client
        client = ElevenLabs(api_key=api_key or ELEVENLABS_API_KEY)
        
        # Convert text to speech using the client API
        audio_bytes = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
            if isinstance(audio_bytes, bytes):
                tmp_audio.write(audio_bytes)
            else:  # Handle streaming response if returned
                for chunk in audio_bytes:
                    if chunk:
                        tmp_audio.write(chunk)
            audio_path = tmp_audio.name
            
        return audio_path, None
    except Exception as e:
        return None, str(e)
