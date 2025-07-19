import streamlit as st
from src.dynamic_workflow import build_workflow, AgentState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
import os
import json
import pprint

st.set_page_config(page_title="Talent Talk", layout="wide")
st.title("Talent Talk")
st.markdown("*AI-powered technical interviews with dynamic resume analysis*")

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

# --- Main App: Interview Loop ---
st.header("Interview")
user_input = st.text_input("Your answer (as candidate):", "")
if st.button("Send") and user_input:
    # Add the human message to the state
    st.session_state.state["messages"].append(HumanMessage(content=user_input))
    
    # Check if the last AI message contains "that's it for today" to detect end of interview
    interview_ended = False
    if st.session_state.state["messages"]:
        for msg in reversed(st.session_state.state["messages"]):
            if isinstance(msg, AIMessage) and "that's it for today" in msg.content.lower():
                interview_ended = True
                st.info("Interview has ended. Generating evaluation and report...")
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
        resume_path=st.session_state.state.get("resume_path"),  # Include the resume path
        questions_path=st.session_state.state.get("questions_path")  # Include the questions path
    )
    
    try:
        # Run the workflow step
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
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

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
    
    # Display PDF download button if PDF path is available
    if st.session_state.state.get("pdf_path") and os.path.exists(st.session_state.state["pdf_path"]):
        with open(st.session_state.state["pdf_path"], "rb") as f:
            st.download_button(
                "Download PDF Report", 
                f, 
                file_name=os.path.basename(st.session_state.state["pdf_path"])
            )

# --- Display App State ---
st.sidebar.markdown("---")
display_app_state()

# --- Footer ---
st.markdown("---")
st.markdown("""
**Talent Talk** | Revolutionizing technical interviews with AI

**Powered by:**
- **LangGraph** - AI Agent
- **Google Generative AI** - Language Model
- **Streamlit** - Web Interface
""")