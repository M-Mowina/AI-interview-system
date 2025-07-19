'''
+-------------------+        +-----------------------+        +------------------+        +------------------------+
|   Step 1: Install |        |  Step 2: Real-Time    |        |  Step 3: Pass    |        |  Step 4: Live Audio    |
|   Python Libraries|        |  Transcription with   |        |  Real-Time       |        |  Stream from ElevenLabs|
+-------------------+        |       AssemblyAI      |        |  Transcript to   |        |                        |
|                   |        +-----------------------+        |      OpenAI      |        +------------------------+
| - assemblyai      |                    |                    +------------------+                    |
| - openai          |                    |                             |                              |
| - elevenlabs      |                    v                             v                              v
| - mpv             |        +-----------------------+        +------------------+        +------------------------+
| - portaudio       |        |                       |        |                  |        |                        |
+-------------------+        |  AssemblyAI performs  |-------->  OpenAI generates|-------->  ElevenLabs streams   |
                             |  real-time speech-to- |        |  response based  |        |  response as live      |
                             |  text transcription   |        |  on transcription|        |  audio to the user     |
                             |                       |        |                  |        |                        |
                             +-----------------------+        +------------------+        +------------------------+

###### Step 1: Install Python libraries ######

brew install portaudio
pip install "assemblyai[extras]"
pip install elevenlabs==0.3.0b0
brew install mpv
pip install --upgrade openai
'''

import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)
from typing import Type
from elevenlabs import generate, stream

# --- Import LangGraph agent ---
from src.beta_app import app as langgraph_app, AgentState, init_state
from langchain_core.messages import HumanMessage
import os

class AI_Assistant:
    def __init__(self):
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.transcriber = None
        # LangGraph agent state
        self.agent_state = init_state.copy()
        # Prompt
        self.full_transcript = [
            {"role":"system", "content":"You are a receptionist at a dental clinic. Be resourceful and efficient."},
        ]

###### Step 2: Real-Time Transcription with AssemblyAI ######
        
    def start_transcription(self):
        # Define event handlers as methods or functions
        def on_begin(self_client: Type[StreamingClient], event: BeginEvent):
            print(f"Session started: {event.id}")

        def on_turn(self_client: Type[StreamingClient], event: TurnEvent):
            print(f"{event.transcript} ({event.end_of_turn})")
            if event.end_of_turn:
                # Call AI response logic when a full utterance is received
                class DummyTranscript:
                    text = event.transcript
                self.generate_ai_response(DummyTranscript())

        def on_terminated(self_client: Type[StreamingClient], event: TerminationEvent):
            print(f"Session terminated: {event.audio_duration_seconds} seconds of audio processed")

        def on_error(self_client: Type[StreamingClient], error: StreamingError):
            print(f"Error occurred: {error}")

        # Create the streaming client
        client = StreamingClient(
            StreamingClientOptions(
                api_key=os.getenv("ASSEMBLYAI_API_KEY"),
                api_host="streaming.assemblyai.com",
            )
        )

        # Register event handlers
        client.on(StreamingEvents.Begin, on_begin)
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Termination, on_terminated)
        client.on(StreamingEvents.Error, on_error)

        # Connect and start streaming
        client.connect(
            StreamingParameters(
                sample_rate=16000,
                format_turns=True,
            )
        )

        try:
            microphone_stream = aai.extras.MicrophoneStream(sample_rate=16000)
            client.stream(microphone_stream)
        finally:
            client.disconnect(terminate=True)
    
    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print("Session ID:", session_opened.session_id)
        return


    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            self.generate_ai_response(transcript)
        else:
            print(transcript.text, end="\r")


    def on_error(self, error: aai.RealtimeError):
        print("An error occured:", error)
        return


    def on_close(self):
        #print("Closing Session")
        return

###### Step 3: Pass real-time transcript to OpenAI ######
    
    def generate_ai_response(self, transcript):
        self.stop_transcription()
        # Add user message to LangGraph agent state
        self.agent_state["messages"].append(HumanMessage(content=transcript.text))
        print(f"\nPatient: {transcript.text}", end="\r\n")
        # Call LangGraph agent
        result = langgraph_app.invoke(self.agent_state)
        self.agent_state = result
        ai_message = result["messages"][-1]
        ai_text = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
        self.generate_audio(ai_text)
        self.start_transcription()
        print(f"\nReal-time transcription: ", end="\r\n")


###### Step 4: Generate audio with ElevenLabs ######
        
    def generate_audio(self, text):

        self.full_transcript.append({"role":"assistant", "content": text})
        print(f"\nAI Receptionist: {text}")

        audio_stream = generate(
            api_key = self.elevenlabs_api_key,
            text = text,
            voice = "JBFqnCBsd6RMkjVDRZzb",
            stream = True
        )

        stream(audio_stream)

greeting = "Hello! I'm your interviewer from OpenAI. Let's get started. Could you please introduce yourself?"
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)
ai_assistant.start_transcription()