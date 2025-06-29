import streamlit as st
from src.interview_workflow_simple import workflow

st.set_page_config(page_title="AI Interview System", layout="centered")

# Initialize session state
if "workflow" not in st.session_state:
    st.session_state.workflow = workflow
    st.session_state.step = "persona"
    st.session_state.current_question_index = 0

# Main app
st.title("🤖 AI Interview System")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Interview Progress")
steps = ["Persona", "Questions", "Interview", "Report"]
current_step_idx = ["persona", "questions", "interview", "report"].index(st.session_state.step)
st.sidebar.progress((current_step_idx + 1) / len(steps))

# Step 1: Persona Generation
if st.session_state.step == "persona":
    st.header("Step 1: Generate Interviewer Persona")
    st.write("Create a professional interviewer persona for the role.")
    
    col1, col2 = st.columns(2)
    with col1:
        role = st.text_input("Role", value="Data Scientist")
    with col2:
        company = st.text_input("Company", value="TechCorp")
    
    if st.button("Generate Persona"):
        with st.spinner("Generating persona..."):
            st.session_state.workflow.state["role"] = role
            st.session_state.workflow.state["company"] = company
            persona = st.session_state.workflow.generate_persona()
            st.session_state.step = "questions"
            st.experimental_rerun()

# Step 2: Load Questions
elif st.session_state.step == "questions":
    st.header("Step 2: Load Interview Questions")
    st.write("Loading relevant questions for the role.")
    
    if st.button("Load Questions"):
        with st.spinner("Loading questions..."):
            questions = st.session_state.workflow.get_questions()
            st.session_state.step = "interview"
            st.experimental_rerun()

# Step 3: Interview
elif st.session_state.step == "interview":
    st.header("Step 3: Conduct Interview")
    
    # Show current progress
    total_questions = len(st.session_state.workflow.state["questions"])
    answered_questions = len(st.session_state.workflow.state["responses"])
    
    st.write(f"Question {answered_questions + 1} of {total_questions}")
    st.progress((answered_questions + 1) / total_questions)
    
    # Get current question
    if answered_questions < total_questions:
        current_question = st.session_state.workflow.ask_question(answered_questions)
        
        st.subheader("Question:")
        st.write(f"**{current_question}**")
        
        # Response input
        user_response = st.text_area("Your Response:", height=150)
        
        if st.button("Submit Response"):
            if user_response.strip():
                with st.spinner("Evaluating response..."):
                    eval_result = st.session_state.workflow.evaluate_response(user_response)
                    
                    # Show evaluation
                    st.success("Response evaluated!")
                    st.write(f"**Score:** {eval_result['overall_score']}/10")
                    st.write(f"**Feedback:** {eval_result['feedback']}")
                    
                    # Move to next question or finish
                    if answered_questions + 1 >= total_questions:
                        st.session_state.step = "report"
                        st.experimental_rerun()
                    else:
                        st.session_state.current_question_index = answered_questions + 1
                        st.experimental_rerun()
            else:
                st.error("Please provide a response.")
    
    # Show previous responses and evaluations
    if st.session_state.workflow.state["responses"]:
        st.subheader("Previous Responses:")
        for i, (response, eval_result) in enumerate(zip(
            st.session_state.workflow.state["responses"],
            st.session_state.workflow.state["evaluations"]
        )):
            with st.expander(f"Question {i+1} - Score: {eval_result['overall_score']}/10"):
                st.write(f"**Question:** {st.session_state.workflow.state['questions'][i]}")
                st.write(f"**Response:** {response}")
                st.write(f"**Feedback:** {eval_result['feedback']}")

# Step 4: Report
elif st.session_state.step == "report":
    st.header("Step 4: Interview Report")
    
    if st.session_state.workflow.state["report"] is None:
        with st.spinner("Generating report..."):
            report = st.session_state.workflow.generate_report()
    
    if st.session_state.workflow.state["report"]:
        report = st.session_state.workflow.state["report"]
        
        st.success("Interview completed! Here's your report:")
        
        # Executive Summary
        st.subheader("Executive Summary")
        st.write(report["executive_summary"])
        
        # Session Score
        session_score = report['session_evaluation']['session_score']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Session Score", f"{session_score}/10")
        with col2:
            st.metric("Total Responses", report['session_evaluation']['total_responses'])
        with col3:
            st.metric("Excellent Responses", report['session_evaluation']['excellent_responses'])
        
        # Detailed Analysis
        st.subheader("Detailed Analysis")
        st.write(report["detailed_analysis"])
        
        # Recommendations
        st.subheader("Recommendations")
        st.write(report["recommendations"])
        
        # Question-by-Question Breakdown
        st.subheader("Question-by-Question Breakdown")
        for i, eval_result in enumerate(st.session_state.workflow.state["evaluations"]):
            with st.expander(f"Question {i+1} - Score: {eval_result['overall_score']}/10"):
                st.write(f"**Question:** {st.session_state.workflow.state['questions'][i]}")
                st.write(f"**Response:** {st.session_state.workflow.state['responses'][i]}")
                st.write(f"**Feedback:** {eval_result['feedback']}")
                if 'llm_evaluation' in eval_result:
                    llm_eval = eval_result['llm_evaluation']
                    if 'technical_correctness' in llm_eval:
                        st.write(f"**Technical Score:** {llm_eval['technical_correctness']['score']}/10")
                    if 'communication_skills' in llm_eval:
                        st.write(f"**Communication Score:** {llm_eval['communication_skills']['score']}/10")
        
        # Reset button
        if st.button("Start New Interview"):
            st.session_state.workflow.reset()
            st.session_state.step = "persona"
            st.session_state.current_question_index = 0
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("*Powered by Google Gemini AI and Streamlit*") 