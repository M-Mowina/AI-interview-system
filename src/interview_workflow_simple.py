"""
Simplified LangGraph Workflow for AI Interview System
Orchestrates Persona Generator, Question Manager, Speech Manager, Response Evaluator, and Report Generator.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, List

# Import agents
from src.agents.persona_generator import PersonaGenerator
from src.agents.question_manager_simple import QuestionManager
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

class InterviewWorkflow:
    def __init__(self):
        self.state = {
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
    
    def generate_persona(self):
        """Step 1: Generate interviewer persona"""
        persona = persona_agent.generate_persona(self.state["company"], self.state["role"])
        self.state["persona"] = persona
        return persona
    
    def get_questions(self):
        """Step 2: Get questions for the role"""
        questions = question_agent.get_questions_by_role(self.state["role"], limit=5)
        self.state["questions"] = questions
        return questions
    
    def ask_question(self, question_index: int):
        """Step 3: Ask a specific question"""
        if question_index < len(self.state["questions"]):
            question = self.state["questions"][question_index]
            self.state["current_question"] = question
            return question
        return None
    
    def evaluate_response(self, response: str):
        """Step 4: Evaluate the candidate's response"""
        if self.state["current_question"] and response:
            eval_result = evaluator.evaluate_response(
                self.state["current_question"],
                response,
                self.state["role"],
                "technical"
            )
            self.state["responses"].append(response)
            self.state["evaluations"].append(eval_result)
            return eval_result
        return None
    
    def generate_report(self):
        """Step 5: Generate final report"""
        if self.state["evaluations"]:
            session_eval = evaluator.evaluate_interview_session(self.state["evaluations"])
            self.state["session_evaluation"] = session_eval
            report = report_agent.generate_interview_report(
                self.state["candidate_info"],
                session_eval,
                self.state["evaluations"],
                self.state["role"],
                self.state["company"]
            )
            self.state["report"] = report
            return report
        return None
    
    def get_state(self):
        """Get current workflow state"""
        return self.state
    
    def reset(self):
        """Reset the workflow state"""
        self.state = {
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

# Create a global workflow instance
workflow = InterviewWorkflow()

# Export functions for Streamlit
def generate_persona_node(state):
    """Node function for persona generation"""
    persona = workflow.generate_persona()
    state["persona"] = persona
    return state

def get_questions_node(state):
    """Node function for getting questions"""
    questions = workflow.get_questions()
    state["questions"] = questions
    return state

def ask_question_node(state):
    """Node function for asking questions"""
    idx = len(state.get("responses", []))
    question = workflow.ask_question(idx)
    state["current_question"] = question
    return state

def evaluate_response_node(state):
    """Node function for evaluating responses"""
    response = state.get("current_response", "")
    if response:
        eval_result = workflow.evaluate_response(response)
        if eval_result:
            if "responses" not in state:
                state["responses"] = []
            if "evaluations" not in state:
                state["evaluations"] = []
            state["responses"].append(response)
            state["evaluations"].append(eval_result)
    return state

def generate_report_node(state):
    """Node function for generating reports"""
    report = workflow.generate_report()
    state["report"] = report
    return state

def initial_state():
    """Get initial state"""
    return workflow.get_state()

if __name__ == "__main__":
    # Test the workflow
    print("Testing Interview Workflow...")
    
    # Step 1: Generate persona
    persona = workflow.generate_persona()
    print(f"Persona: {persona[:100]}...")
    
    # Step 2: Get questions
    questions = workflow.get_questions()
    print(f"Questions: {len(questions)} questions loaded")
    
    # Step 3: Ask first question
    question = workflow.ask_question(0)
    print(f"Question: {question}")
    
    # Step 4: Evaluate response
    response = "I have experience with machine learning projects including classification and regression tasks."
    eval_result = workflow.evaluate_response(response)
    print(f"Evaluation: {eval_result['overall_score']}/10")
    
    # Step 5: Generate report
    report = workflow.generate_report()
    print(f"Report generated: {report['report_id']}") 