import streamlit as st
from src.workflow import build_workflow, AgentState
from langchain_core.messages import HumanMessage, AIMessage
import os

st.set_page_config(page_title="Alpha Interview System", layout="wide")
st.title("Alpha AI Interview System")

# --- Sidebar: Resume Upload and Interview Setup ---
st.sidebar.header("Candidate Setup")
resume_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])
mode = st.sidebar.selectbox("Interviewer Mode", ["friendly", "formal", "technical"])
position = st.sidebar.text_input("Position", "AI Developer")
company = st.sidebar.text_input("Company Name", "Tech Innovators Inc.")
num_of_q = st.sidebar.number_input("Number of Technical Questions", min_value=1, max_value=10, value=2)
num_of_follow_up = st.sidebar.number_input("Number of Follow-up Questions", min_value=0, max_value=3, value=1)

# --- Save uploaded resume ---
resume_path = None
if resume_file:
    resume_dir = "./uploaded_resumes"
    os.makedirs(resume_dir, exist_ok=True)
    resume_path = os.path.join(resume_dir, resume_file.name)
    with open(resume_path, "wb") as f:
        f.write(resume_file.read())
    st.sidebar.success(f"Resume uploaded: {resume_file.name}")

# --- Initialize workflow ---
workflow = build_workflow()

# --- Session State ---
if "state" not in st.session_state:
    # Initialize with empty state
    st.session_state.state = {
        "mode": mode,
        "num_of_q": num_of_q,
        "num_of_follow_up": num_of_follow_up,
        "position": position,
        "evaluation_result": "",  # Initialize with empty string for add operator
        "company_name": company,
        "messages": [],  # Will be handled by add_messages
        "report": "",  # Initialize with empty string for add operator
        "pdf_path": None,
    }

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
                break
    
    # Create a fresh state dictionary for the workflow
    current_state = {
        "mode": st.session_state.state["mode"],
        "num_of_q": st.session_state.state["num_of_q"],
        "num_of_follow_up": st.session_state.state["num_of_follow_up"],
        "position": st.session_state.state["position"],
        "company_name": st.session_state.state["company_name"],
        "messages": st.session_state.state["messages"],
        "pdf_path": st.session_state.state["pdf_path"]
    }
    
    # Only include evaluation_result and report if they exist and we're not at the end of the interview
    if not interview_ended:
        # For normal conversation, we want to keep the existing evaluation and report
        current_state["evaluation_result"] = st.session_state.state.get("evaluation_result", "")
        current_state["report"] = st.session_state.state.get("report", "")
    else:
        # For end of interview, we want to reset the evaluation and report
        # This ensures we get fresh evaluation and report
        current_state["evaluation_result"] = ""
        current_state["report"] = ""
    
    try:
        # Run the workflow step
        result = workflow.invoke(current_state)
        
        # Special handling for evaluation_result to avoid concatenation issues
        if "evaluation_result" in result and result["evaluation_result"]:
            # If we're at the end of the interview, replace the evaluation result
            if interview_ended:
                st.session_state.state["evaluation_result"] = result["evaluation_result"]
            # Otherwise, keep the existing evaluation result
            else:
                pass  # Don't update evaluation_result during normal conversation
        
        # Special handling for report to avoid concatenation issues
        if "report" in result and result["report"]:
            # Always replace the report
            st.session_state.state["report"] = result["report"]
        
        # Update other fields normally
        for key, value in result.items():
            if key not in ["evaluation_result", "report"]:
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

# --- Display Evaluation ---
if st.session_state.state.get("evaluation_result"):
    st.subheader("Evaluation Result")
    st.markdown(st.session_state.state["evaluation_result"])

# --- Display Report and PDF Download ---
if st.session_state.state.get("report"):
    st.subheader("HR Report")
    st.markdown(st.session_state.state["report"])
    if st.session_state.state.get("pdf_path") and os.path.exists(st.session_state.state["pdf_path"]):
        with open(st.session_state.state["pdf_path"], "rb") as f:
            st.download_button("Download PDF Report", f, file_name="interview_report.pdf") 