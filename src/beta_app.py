import os
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
from workflow import build_workflow, AgentState

# =========================
# Configuration & Setup
# =========================
load_dotenv()

# Initialize the workflow
workflow = build_workflow()

# Initial agent state (customize as needed)
init_state = AgentState(
    mode="friendly",
    num_of_q=1,
    num_of_follow_up=0,
    position="AI Developer",
    evaluation_result="",
    company_name="OpenAI",
    messages=[],
    report="",
    pdf_path=""
)

# =========================
# Chat Loop
# =========================
def chat_loop(initial_state: AgentState):
    """
    Runs a conversational loop with the new workflow (Q&A, evaluation, report, PDF).
    """
    current_state = initial_state
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        current_state["messages"].append(HumanMessage(content=user_input))
        # Run the workflow step
        result = workflow.invoke(current_state)
        current_state.update(result)
        # Print the AI's response
        ai_message = current_state["messages"][-1]
        if isinstance(ai_message, AIMessage):
            print(f"AI Recruiter:\n{ai_message.content}")
        elif isinstance(ai_message, ToolMessage):
            print(f"AI Recruiter (Tool Output):\n{ai_message.content}")
        else:
            print(f"AI Recruiter (Other Message Type): \n{ai_message}")
        # If evaluation or report is available, print it
        if current_state.get("evaluation_result"):
            print("\n--- Evaluation Result ---")
            print(current_state["evaluation_result"])
        if current_state.get("report"):
            print("\n--- HR Report ---")
            print(current_state["report"])
        if current_state.get("pdf_path"):
            print(f"\nPDF report saved at: {current_state['pdf_path']}")
    return current_state

# =========================
# Main Entry Point
# =========================
if __name__ == "__main__":
    final_state = chat_loop(init_state)