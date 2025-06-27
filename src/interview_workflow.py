"""
LangGraph Workflow for AI Interview System
Orchestrates Persona Generator, Question Manager, Speech Manager, Response Evaluator, and Report Generator.
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Import agents
from src.agents.persona_generator import PersonaGenerator
from src.agents.question_manager import QuestionManager
from src.agents.speech_manager import SpeechManager
from src.agents.response_evaluator import ResponseEvaluator
from src.agents.report_generator import ReportGenerator

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize agents
persona_agent = PersonaGenerator(GOOGLE_API_KEY)
question_agent = QuestionManager(GOOGLE_API_KEY)
speech_agent = SpeechManager()
evaluator = ResponseEvaluator(GOOGLE_API_KEY)
report_agent = ReportGenerator(GOOGLE_API_KEY)

# Define the state for the workflow
def initial_state():
    return {
        "persona": None,
        "questions": [],
        "current_question": None,
        "responses": [],
        "evaluations": [],
        "session_evaluation": None,
        "report": None,
        "candidate_info": {},
        "role": "Data Scientist",
        "company": "TechCorp"
    }

# Node: Generate Persona
def generate_persona_node(state):
    persona = persona_agent.generate_persona(state["company"], state["role"])
    state["persona"] = persona
    return state

# Node: Get Questions
def get_questions_node(state):
    questions = question_agent.get_questions_by_role(state["role"], limit=5)
    state["questions"] = questions
    return state

# Node: Ask Question (simulate speech input/output)
def ask_question_node(state):
    idx = len(state["responses"])
    if idx < len(state["questions"]):
        question = state["questions"][idx]
        print(f"AI: {question}")
        # Simulate speech input (replace with speech_agent.speech_to_text() for real use)
        user_response = input("You: ")
        state["current_question"] = question
        state["current_response"] = user_response
    else:
        state["current_question"] = None
        state["current_response"] = None
    return state

# Node: Evaluate Response
def evaluate_response_node(state):
    if state["current_question"] and state["current_response"]:
        eval_result = evaluator.evaluate_response(
            state["current_question"],
            state["current_response"],
            state["role"],
            "technical"  # For demo; could be dynamic
        )
        state["responses"].append(state["current_response"])
        state["evaluations"].append(eval_result)
    return state

# Node: Check if more questions
def check_continue_node(state):
    if len(state["responses"]) < len(state["questions"]):
        return "ask_question"
    else:
        return "generate_report"

# Node: Generate Report
def generate_report_node(state):
    session_eval = evaluator.evaluate_interview_session(state["evaluations"])
    state["session_evaluation"] = session_eval
    report = report_agent.generate_interview_report(
        state["candidate_info"],
        session_eval,
        state["evaluations"],
        state["role"],
        state["company"]
    )
    state["report"] = report
    print("\n--- INTERVIEW REPORT ---\n")
    print(report["executive_summary"])
    return state

# Build the LangGraph workflow
graph = StateGraph(initial_state)
graph.add_node("generate_persona", generate_persona_node)
graph.add_node("get_questions", get_questions_node)
graph.add_node("ask_question", ask_question_node)
graph.add_node("evaluate_response", evaluate_response_node)
graph.add_node("check_continue", check_continue_node)
graph.add_node("generate_report", generate_report_node)

graph.set_entry_point("generate_persona")
graph.add_edge("generate_persona", "get_questions")
graph.add_edge("get_questions", "ask_question")
graph.add_edge("ask_question", "evaluate_response")
graph.add_edge("evaluate_response", "check_continue")
graph.add_conditional_edges(
    "check_continue",
    {
        "ask_question": "ask_question",
        "generate_report": "generate_report"
    }
)
graph.add_edge("generate_report", END)

workflow = graph.compile()

if __name__ == "__main__":
    # Run the workflow interactively
    state = workflow.run() 