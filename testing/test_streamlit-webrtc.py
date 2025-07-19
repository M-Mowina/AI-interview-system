import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        return frame

webrtc_streamer(key="audio", audio_processor_factory=AudioProcessor)