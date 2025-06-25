"""
AI Interview System - app.py
This is the entry point for the AI Interview System project.
Currently under development.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Uncomment these for voice features in the future
# import speech_recognition as sr
# import pyttsx3
# from gtts import gTTS

ROLE_PROMPT = """
You are an AI interviewer for a leading tech company, conducting an interview for a Data Scientist position.

Your goal is to assess the candidate's technical skills, problem-solving abilities, communication skills, and experience relevant to data science roles.

Maintain a professional yet approachable tone. Start by introducing yourself as the interviewer and asking the candidate to introduce themselves and walk through their resume.

Focus on questions related to:
- Machine Learning concepts and algorithms
- Statistical analysis and probability
- Data manipulation and cleaning (Python, SQL)
- Model evaluation and deployment
- Problem-solving and case studies
- Relevant projects and past experiences

Begin the interview now.
"""

def main():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        return

    client = genai.Client(api_key=google_api_key)
    chat = client.chats.create(
        model='gemini-2.5-flash-lite-preview-06-17',
        config=types.GenerateContentConfig(
            system_instruction=ROLE_PROMPT),
    )

    print("Welcome to the AI Interview System! Type 'exit' to quit.\n")
    # Initial greeting from the AI
    response = chat.send_message("Hi")
    print(f"AI: {response.text}\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Exiting interview. Goodbye!")
            break
        response = chat.send_message(user_input)
        print(f"AI: {response.text}\n")
        # TODO: Add TTS (text-to-speech) for AI responses
        # TODO: Add voice input option

if __name__ == "__main__":
    main() 