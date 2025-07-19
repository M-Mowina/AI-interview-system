import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import wave
import tempfile
import time
from audio_recorder_streamlit import audio_recorder
from utils.audio_utils import transcribe_audio_file, elevenlabs_tts
from src.dynamic_workflow import build_workflow, AgentState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if AssemblyAI API key is set
if not os.getenv("ASSEMBLYAI_API_KEY"):
    st.error("⚠️ AssemblyAI API key not found. Please set the ASSEMBLYAI_API_KEY environment variable.")

# Check if ElevenLabs API key is set
if not os.getenv("ELEVENLABS_API_KEY"):
    st.warning("⚠️ ElevenLabs API key not found. Voice responses will not be available.")

st.set_page_config(page_title="Talent Talk", layout="wide")
st.title("Talent Talk")
st.markdown("*AI-powered technical interviews with cloud-based voice interaction*")

# Function to display the app state
def display_app_state():
    """Display the current application state in a simple way"""
    with st.expander("App State", expanded=False):
        # Convert messages to strings for display
        state_copy = dict(st.session_state.state)
        
        # Handle messages separately
        if "messages" in state_copy:
            messages_str = []
            for i, msg in enumerate(state_copy["messages"]):
                msg_type = type(msg).__name__
                if hasattr(msg, "content"):
                    content = f"{msg_type}: {msg.content}"
                else:
                    content = f"{msg_type}: {str(msg)}"
                messages_str.append(content)
            
            # Replace messages with string representation
            state_copy["messages"] = messages_str
        
        # Display the state
        st.code(str(state_copy), language="python")

# --- Sidebar: Interview Setup ---
st.sidebar.header("Interview Setup")

# Get current values from session state if available
current_mode = st.session_state.state.get("mode", "friendly") if "state" in st.session_state else "friendly"
current_position = st.session_state.state.get("position", "AI Developer") if "state" in st.session_state else "AI Developer"
current_company = st.session_state.state.get("company_name", "Tech Innovators Inc.") if "state" in st.session_state else "Tech Innovators Inc."
current_num_q = st.session_state.state.get("num_of_q", 2) if "state" in st.session_state else 2
current_num_follow = st.session_state.state.get("num_of_follow_up", 1) if "state" in st.session_state else 1

# Interview configuration
mode = st.sidebar.selectbox("Interviewer Mode", ["friendly", "formal", "technical"], index=["friendly", "formal", "technical"].index(current_mode))
position = st.sidebar.text_input("Position", value=current_position)
company = st.sidebar.text_input("Company Name", value=current_company)
num_of_q = st.sidebar.number_input("Number of Technical Questions", min_value=1, max_value=10, value=current_num_q)
num_of_follow_up = st.sidebar.number_input("Number of Follow-up Questions", min_value=0, max_value=3, value=current_num_follow)

# Button to update interview parameters
params_changed = (
    "state" in st.session_state and (
        mode != st.session_state.state.get("mode") or
        position != st.session_state.state.get("position") or
        company != st.session_state.state.get("company_name") or
        num_of_q != st.session_state.state.get("num_of_q") or
        num_of_follow_up != st.session_state.state.get("num_of_follow_up")
    )
)

if params_changed:
    st.sidebar.warning("Interview parameters have changed. Click 'Update Parameters' to apply changes.")
    
update_params = st.sidebar.button("Update Parameters")
if update_params and "state" in st.session_state:
    st.session_state.state["mode"] = mode
    st.session_state.state["position"] = position
    st.session_state.state["company_name"] = company
    st.session_state.state["num_of_q"] = num_of_q
    st.session_state.state["num_of_follow_up"] = num_of_follow_up
    st.sidebar.success("Interview parameters updated!")

# File uploads section
st.sidebar.header("Upload Files")

# Resume upload
st.sidebar.subheader("Candidate Resume")
resume_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume_uploader")
resume_path = None
if resume_file:
    resume_dir = "./uploaded_resumes"
    os.makedirs(resume_dir, exist_ok=True)
    resume_path = os.path.join(resume_dir, resume_file.name)
    with open(resume_path, "wb") as f:
        f.write(resume_file.read())
    st.sidebar.success(f"Resume uploaded: {resume_file.name}")

# Interview questions upload
st.sidebar.subheader("Interview Questions")
questions_file = st.sidebar.file_uploader("Upload Questions (PDF)", type=["pdf"], key="questions_uploader")
questions_path = None
if questions_file:
    questions_dir = "./uploaded_questions"
    os.makedirs(questions_dir, exist_ok=True)
    questions_path = os.path.join(questions_dir, questions_file.name)
    with open(questions_path, "wb") as f:
        f.write(questions_file.read())
    st.sidebar.success(f"Questions uploaded: {questions_file.name}")

# Voice settings
st.sidebar.header("Voice Settings")
input_method = st.sidebar.radio("Input Method", ["Text", "Voice"], index=0)

# --- Initialize workflow ---
workflow = build_workflow()

# --- Session State ---
if "state" not in st.session_state:
    # Initialize with empty state
    st.session_state.state = AgentState(
        mode=mode,
        num_of_q=num_of_q,
        num_of_follow_up=num_of_follow_up,
        position=position,
        evaluation_result="",
        company_name=company,
        messages=[],
        report="",
        pdf_path=None,
        resume_path=resume_path,  # Include the resume path in the state
        questions_path=questions_path  # Include the questions path in the state
    )
# Update paths if they change
else:
    # Update resume path if it changes
    if resume_path and st.session_state.state.get("resume_path") != resume_path:
        st.session_state.state["resume_path"] = resume_path
        st.sidebar.info("Resume updated. The interviewer will use your new resume.")
    
    # Update questions path if it changes
    if questions_path and st.session_state.state.get("questions_path") != questions_path:
        st.session_state.state["questions_path"] = questions_path
        st.sidebar.info("Interview questions updated. The interviewer will use your new questions.")

# Function to process messages (both text and voice)
def process_message(user_input):
    # Check if the last AI message contains "that's it for today" to detect end of interview
    interview_ended = False
    if st.session_state.state["messages"]:
        for msg in reversed(st.session_state.state["messages"]):
            if isinstance(msg, AIMessage) and "that's it for today" in msg.content.lower():
                interview_ended = True
                st.info("Interview has ended. You can generate the evaluation and report.")
                break
    
    # Create a fresh state dictionary for the workflow
    current_state = AgentState(
        mode=st.session_state.state["mode"],
        num_of_q=st.session_state.state["num_of_q"],
        num_of_follow_up=st.session_state.state["num_of_follow_up"],
        position=st.session_state.state["position"],
        company_name=st.session_state.state["company_name"],
        messages=st.session_state.state["messages"],
        evaluation_result="" if interview_ended else st.session_state.state.get("evaluation_result", ""),
        report="" if interview_ended else st.session_state.state.get("report", ""),
        pdf_path=st.session_state.state.get("pdf_path"),
        resume_path=st.session_state.state.get("resume_path"),
        questions_path=st.session_state.state.get("questions_path")
    )
    
    try:
        # Run the workflow step
        with st.spinner("AI is thinking..."):
            result = workflow.invoke(current_state)
        
        # Update session state with the result
        for key, value in result.items():
            if key == "evaluation_result" and interview_ended:
                # Replace evaluation result at the end of interview
                st.session_state.state["evaluation_result"] = value
            elif key == "report" and value:
                # Always replace the report with new content
                st.session_state.state["report"] = value
            elif key == "pdf_path" and value:
                # Update PDF path when available
                st.session_state.state["pdf_path"] = value
            elif key == "messages":
                # Messages are handled by add_messages
                st.session_state.state["messages"] = value
            else:
                # Update other fields normally
                st.session_state.state[key] = value
        
        # Get the AI's response
        ai_message = st.session_state.state["messages"][-1]
        if isinstance(ai_message, AIMessage):
            ai_text = ai_message.content
            
            # Display the AI's text response
            st.subheader("AI Response")
            st.write(ai_text)
            
            # Generate and play voice response if using voice input
            if input_method == "Voice" and os.getenv("ELEVENLABS_API_KEY"):
                with st.spinner("Generating voice response..."):
                    audio_path, tts_error = elevenlabs_tts(ai_text, os.getenv("ELEVENLABS_API_KEY"))
                    if tts_error:
                        st.error(f"Voice generation failed: {tts_error}")
                    elif audio_path:
                        st.subheader("AI Voice Response")
                        st.audio(audio_path, format="audio/mp3")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

# --- Main App: Interview Loop ---
st.header("Interview")

# Voice input
if input_method == "Voice":
    MAX_RECORD_SECONDS = 30
    st.info(f"Please keep your recording under {MAX_RECORD_SECONDS} seconds for best results.")
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
            with st.status("Transcribing audio with AssemblyAI...", expanded=True) as status:
                transcribed_text, stt_error = transcribe_audio_file(audio_path)
                if stt_error:
                    status.update(label="Transcription failed", state="error")
                    st.error(stt_error)
                elif transcribed_text:
                    status.update(label="Transcription complete!", state="complete")
                    st.subheader("Transcription")
                    st.write(transcribed_text)
                    
                    # Add the human message to the state
                    st.session_state.state["messages"].append(HumanMessage(content=transcribed_text))
                    
                    # Process the message
                    process_message(transcribed_text)
else:
    # Text input
    user_input = st.text_input("Your answer (as candidate):", "")
    if st.button("Send") and user_input:
        # Add the human message to the state
        st.session_state.state["messages"].append(HumanMessage(content=user_input))
        
        # Process the message
        process_message(user_input)

# --- Display Transcript ---
st.subheader("Transcript")
for m in st.session_state.state["messages"]:
    if isinstance(m, HumanMessage):
        st.markdown(f"**Candidate:** {m.content}")
    elif isinstance(m, AIMessage):
        st.markdown(f"**AI Recruiter:** {m.content}")

# Check if interview has ended but evaluation hasn't been generated
interview_ended = False
for msg in reversed(st.session_state.state.get("messages", [])):
    if isinstance(msg, AIMessage) and "that's it for today" in msg.content.lower():
        interview_ended = True
        break

if interview_ended and not st.session_state.state.get("evaluation_result"):
    st.warning("Interview has ended. Click the button below to generate evaluation and report.")
    if st.button("Generate Evaluation and Report"):
        st.info("Generating evaluation and report... This may take a moment.")
        
        try:
            # Create a state with empty evaluation_result to trigger the evaluator
            current_state = AgentState(
                mode=st.session_state.state["mode"],
                num_of_q=st.session_state.state["num_of_q"],
                num_of_follow_up=st.session_state.state["num_of_follow_up"],
                position=st.session_state.state["position"],
                company_name=st.session_state.state["company_name"],
                messages=st.session_state.state["messages"],
                evaluation_result="",  # Empty to trigger evaluation
                report="",  # Empty to trigger report generation
                pdf_path=None,
                resume_path=st.session_state.state.get("resume_path"),
                questions_path=st.session_state.state.get("questions_path")
            )
            
            # First, run the evaluator with retry logic
            with st.spinner("Generating evaluation..."):
                from src.dynamic_workflow import evaluator
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        # Add a human message to avoid system-prompt-only issues
                        if not any(isinstance(m, HumanMessage) for m in current_state["messages"]):
                            current_state["messages"].append(HumanMessage(content="Please evaluate the interview."))
                        
                        eval_result = evaluator(current_state)
                        st.session_state.state["evaluation_result"] = eval_result["evaluation_result"]
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            st.error(f"Failed to generate evaluation after {max_retries} attempts: {str(e)}")
                            raise
                        st.warning(f"Retry {retry_count}/{max_retries}: {str(e)}")
                        import time
                        time.sleep(2)  # Wait before retrying
            
            # Then, run the report writer with retry logic
            with st.spinner("Generating report..."):
                from src.dynamic_workflow import report_writer
                current_state["evaluation_result"] = st.session_state.state["evaluation_result"]
                
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        report_result = report_writer(current_state)
                        st.session_state.state["report"] = report_result["report"]
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            st.error(f"Failed to generate report after {max_retries} attempts: {str(e)}")
                            raise
                        st.warning(f"Retry {retry_count}/{max_retries}: {str(e)}")
                        import time
                        time.sleep(2)  # Wait before retrying
            
            # Finally, generate the PDF
            with st.spinner("Generating PDF..."):
                from src.dynamic_workflow import pdf_generator_node
                current_state["report"] = st.session_state.state["report"]
                pdf_result = pdf_generator_node(current_state)
                st.session_state.state["pdf_path"] = pdf_result["pdf_path"]
            
            st.success("Evaluation and report generated successfully!")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# --- Display Evaluation ---
if st.session_state.state.get("evaluation_result"):
    st.subheader("Evaluation Result")
    st.markdown(st.session_state.state["evaluation_result"])

# --- Display Report and PDF Download ---
if st.session_state.state.get("report"):
    st.subheader("HR Report")
    st.markdown(st.session_state.state["report"])
    
    # Display PDF path for debugging
    pdf_path = st.session_state.state.get("pdf_path")
    if pdf_path:
        st.info(f"PDF path: {pdf_path}")
    else:
        st.warning("PDF path is not set. Generating PDF now...")
        # Generate PDF directly if path is not set
        try:
            from src.dynamic_workflow import generate_pdf
            filename = f"HR_Report_{st.session_state.state['company_name']}_{st.session_state.state['position']}.pdf".replace(" ", "_")
            pdf_path = generate_pdf(st.session_state.state["report"], filename=filename)
            st.session_state.state["pdf_path"] = pdf_path
            st.success(f"PDF generated at: {pdf_path}")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
    
    # Display PDF download button if PDF path is available
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
            st.download_button(
                "📥 Download PDF Report", 
                pdf_data, 
                file_name=os.path.basename(pdf_path),
                mime="application/pdf"
            )
        st.success("✅ PDF report is ready for download!")
    elif pdf_path:
        st.error(f"PDF file not found at path: {pdf_path}")
        
        # Try to find the file in the generated_reports directory
        generated_dir = "./generated_reports"
        if os.path.exists(generated_dir):
            pdf_files = [f for f in os.listdir(generated_dir) if f.endswith('.pdf')]
            if pdf_files:
                st.info(f"Found {len(pdf_files)} PDF files in the generated_reports directory:")
                for pdf_file in pdf_files:
                    full_path = os.path.join(generated_dir, pdf_file)
                    with open(full_path, "rb") as f:
                        st.download_button(
                            f"📥 Download {pdf_file}", 
                            f, 
                            file_name=pdf_file,
                            mime="application/pdf"
                        )

# --- Display App State ---
st.sidebar.markdown("---")
display_app_state()

# --- Footer ---
st.markdown("---")
st.markdown("""
**Talent Talk** | Revolutionizing technical interviews with AI

**Powered by:**
- **AssemblyAI** - Speech-to-Text
- **ElevenLabs** - Text-to-Speech  
- **LangGraph** - AI Agent
- **Streamlit** - Web Interface
""")
