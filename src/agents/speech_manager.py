"""
Agent 2: Speech Manager
Converts user's speech to text and speaks back using TTS.
"""

import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import os
import tempfile
import subprocess
from typing import Optional, Tuple
import threading
import queue


class SpeechManager:
    def __init__(self, tts_engine: str = "pyttsx3", voice_id: Optional[str] = None):
        """
        Initialize the Speech Manager.
        
        Args:
            tts_engine: "pyttsx3" for offline or "gtts" for online TTS
            voice_id: Specific voice ID for pyttsx3
        """
        self.tts_engine = tts_engine
        self.voice_id = voice_id
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize TTS engine
        if tts_engine == "pyttsx3":
            self.tts_engine_instance = pyttsx3.init()
            if voice_id:
                self.tts_engine_instance.setProperty('voice', voice_id)
            # Set properties for better quality
            self.tts_engine_instance.setProperty('rate', 150)
            self.tts_engine_instance.setProperty('volume', 0.9)
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
    
    def speech_to_text(self, timeout: int = 5, phrase_time_limit: int = 10) -> Tuple[bool, str]:
        """
        Convert speech to text using Google Speech Recognition.
        
        Args:
            timeout: Time to wait for speech to start
            phrase_time_limit: Maximum time for a single phrase
            
        Returns:
            Tuple of (success, text)
        """
        try:
            with self.microphone as source:
                print("Listening... (speak now)")
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            
            # Use Google Speech Recognition
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return True, text
            
        except sr.WaitTimeoutError:
            print("No speech detected within timeout period")
            return False, ""
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return False, ""
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return False, ""
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return False, ""
    
    def text_to_speech(self, text: str, save_audio: bool = False, filename: str = "response.mp3") -> bool:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            save_audio: Whether to save audio to file
            filename: Filename to save audio (if save_audio is True)
            
        Returns:
            Success status
        """
        try:
            if self.tts_engine == "pyttsx3":
                return self._tts_pyttsx3(text, save_audio, filename)
            elif self.tts_engine == "gtts":
                return self._tts_gtts(text, save_audio, filename)
            else:
                print(f"Unsupported TTS engine: {self.tts_engine}")
                return False
                
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            return False
    
    def _tts_pyttsx3(self, text: str, save_audio: bool, filename: str) -> bool:
        """Use pyttsx3 for text-to-speech."""
        try:
            self.tts_engine_instance.say(text)
            self.tts_engine_instance.runAndWait()
            return True
        except Exception as e:
            print(f"Error with pyttsx3: {e}")
            return False
    
    def _tts_gtts(self, text: str, save_audio: bool, filename: str) -> bool:
        """Use gTTS for text-to-speech."""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            
            if save_audio:
                tts.save(filename)
                # Play the audio file
                self._play_audio_file(filename)
            else:
                # Save to temporary file and play
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tts.save(tmp_file.name)
                    self._play_audio_file(tmp_file.name)
                    os.unlink(tmp_file.name)
            
            return True
            
        except Exception as e:
            print(f"Error with gTTS: {e}")
            return False
    
    def _play_audio_file(self, filename: str):
        """Play audio file using system default player."""
        try:
            if os.name == 'nt':  # Windows
                os.system(f'start {filename}')
            elif os.name == 'posix':  # macOS and Linux
                subprocess.run(['open', filename], check=True)
        except Exception as e:
            print(f"Error playing audio file: {e}")
    
    def continuous_listening(self, callback, stop_event=None):
        """
        Continuously listen for speech and call callback with recognized text.
        
        Args:
            callback: Function to call with recognized text
            stop_event: Threading event to stop listening
        """
        def listen_loop():
            while not stop_event or not stop_event.is_set():
                success, text = self.speech_to_text()
                if success and text:
                    callback(text)
        
        thread = threading.Thread(target=listen_loop)
        thread.daemon = True
        thread.start()
        return thread
    
    def test_microphone(self) -> bool:
        """Test if microphone is working properly."""
        try:
            success, text = self.speech_to_text(timeout=3)
            if success:
                print(f"Microphone test successful! You said: {text}")
                return True
            else:
                print("Microphone test failed - no speech detected")
                return False
        except Exception as e:
            print(f"Microphone test failed: {e}")
            return False
    
    def get_available_voices(self) -> list:
        """Get list of available voices for pyttsx3."""
        if self.tts_engine == "pyttsx3":
            voices = self.tts_engine_instance.getProperty('voices')
            return [voice.id for voice in voices]
        return []
    
    def set_voice(self, voice_id: str):
        """Set specific voice for pyttsx3."""
        if self.tts_engine == "pyttsx3":
            self.tts_engine_instance.setProperty('voice', voice_id)
            self.voice_id = voice_id


if __name__ == "__main__":
    # Test the speech manager
    speech_manager = SpeechManager(tts_engine="pyttsx3")
    
    # Test microphone
    print("Testing microphone...")
    if speech_manager.test_microphone():
        print("Microphone is working!")
        
        # Test TTS
        print("Testing text-to-speech...")
        speech_manager.text_to_speech("Hello! This is a test of the speech manager.")
    else:
        print("Microphone test failed. Please check your microphone settings.") 