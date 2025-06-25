# AI Interview System

This project aims to build an AI-powered interview assistant system. The system will leverage advanced language models to conduct, assess, and provide feedback for technical interviews, starting with Data Scientist roles.

## Status

**This project is currently under development.**

## Getting Started

The main entry point for the project is `app.py`.

---

## Agent Architecture & Logic

The system is composed of five modular agents, each responsible for a key part of the interview workflow:

### 1. Persona Generator
- **Purpose:** Customizes the interviewer persona based on company, role, and preferences.
- **Logic:** Uses prompt engineering (and optionally LLM) to generate a system prompt that defines the interviewer's style, tone, focus areas, and introduction. This ensures the AI interviewer can adapt to different companies and job roles.

### 2. Speech Manager
- **Purpose:** Handles speech-to-text (STT) and text-to-speech (TTS) for voice-based interviews.
- **Logic:** Uses libraries like `speech_recognition` (Google API, Whisper, etc.) for converting user speech to text, and `pyttsx3` or `gTTS` for converting AI responses to audio. Supports both offline and online TTS, and can test microphone and playback.

### 3. Question Manager
- **Purpose:** Selects and manages interview questions tailored to the role and interview stage.
- **Logic:** Uses a vector database (ChromaDB) to store and retrieve questions by role, category, and difficulty. Can generate follow-up questions using an LLM based on candidate responses. Tracks which questions have been asked and manages the flow (introduction, technical, behavioral, etc.).

### 4. Response Evaluator
- **Purpose:** Evaluates candidate answers for quality, correctness, and communication.
- **Logic:** Combines LLM-based evaluation (clarity, confidence, technical correctness, etc.) with rule-based metrics (length, technical terms, specificity). Produces a composite score and detailed feedback for each response.

### 5. Report Generator
- **Purpose:** Summarizes the interview session and generates feedback reports for HR or the candidate.
- **Logic:** Aggregates all evaluations, generates an executive summary and recommendations using the LLM, and can output the report as PDF/HTML/JSON. Supports sending the report via email to HR or the candidate.

---

Stay tuned for updates and more features soon! 
