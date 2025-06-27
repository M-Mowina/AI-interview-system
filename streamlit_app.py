import streamlit as st
from src.interview_workflow import (
    initial_state,
    generate_persona_node,
    get_questions_node,
    ask_question_node,
    evaluate_response_node,
    generate_report_node
)

st.set_page_config(page_title="AI Interview System", layout="centered")
st.title("🤖 AI Interview System")

# Initialize session state for workflow
if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = initial_state()
    st.session_state.step = "persona"
    st.session_state.last_question = None
    st.session_state.last_eval = None

state = st.session_state.workflow_state

# Step 1: Persona Generation
if st.session_state.step == "persona":
    st.header("Step 1: Interviewer Persona")
    st.write(f"**Role:** {state['role']}  |  **Company:** {state['company']}")
    if st.button("Generate Persona"):
        st.session_state.workflow_state = generate_persona_node(state)
        st.session_state.step = "questions"
        st.experimental_rerun()
    if state["persona"]:
        st.success("Persona generated!")
        st.write(state["persona"])

# Step 2: Get Questions
elif st.session_state.step == "questions":
    st.header("Step 2: Interview Questions")
    if st.button("Load Questions"):
        st.session_state.workflow_state = get_questions_node(state)
        st.session_state.step = "interview"
        st.experimental_rerun()
    if state["questions"]:
        st.success(f"Loaded {len(state['questions'])} questions.")
        for i, q in enumerate(state["questions"], 1):
            st.write(f"{i}. {q}")

# Step 3: Interview Q&A Loop
elif st.session_state.step == "interview":
    st.header("Step 3: Interview Q&A")
    idx = len(state["responses"])
    if idx < len(state["questions"]):
        question = state["questions"][idx]
        st.subheader(f"Question {idx+1}")
        st.write(question)
        user_response = st.text_area("Your answer:", key=f"response_{idx}")
        if st.button("Submit Answer", key=f"submit_{idx}"):
            st.session_state.workflow_state["current_question"] = question
            st.session_state.workflow_state["current_response"] = user_response
            st.session_state.workflow_state = evaluate_response_node(st.session_state.workflow_state)
            st.session_state.last_question = question
            st.session_state.last_eval = st.session_state.workflow_state["evaluations"][-1]
            st.experimental_rerun()
        # Show last evaluation if available
        if st.session_state.last_eval and st.session_state.last_question == question:
            st.info(f"**AI Feedback:** {st.session_state.last_eval['feedback']}")
    else:
        st.session_state.step = "report"
        st.experimental_rerun()

# Step 4: Show Report
elif st.session_state.step == "report":
    st.header("Step 4: Interview Report")
    if state["report"] is None:
        st.session_state.workflow_state = generate_report_node(state)
        st.experimental_rerun()
    st.success("Interview complete!")
    st.subheader("Executive Summary")
    st.write(state["report"]["executive_summary"])
    st.subheader("Session Evaluation")
    st.json(state["session_evaluation"])
    st.subheader("Full Report (JSON)")
    st.json(state["report"])
    st.button("Restart", on_click=lambda: st.session_state.clear()) 