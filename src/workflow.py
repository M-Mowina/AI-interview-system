# src/workflow.py
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from operator import add
from langchain_core.tools import tool
import os
#from src.pdf_utils import generate_pdf
from langchain.tools.retriever import create_retriever_tool
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# --- AgentState definition ---
# Using a TypedDict with annotations for LangGraph state management
from typing import Dict, Any, List, TypedDict, Annotated, Sequence
from operator import add
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    mode: str
    num_of_q: int
    num_of_follow_up: int
    position: str
    evaluation_result: Annotated[str, add]
    company_name: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    report: Annotated[str, add]
    pdf_path: str | None

# --- LLM and Embeddings ---
llm = init_chat_model("google_genai:gemini-2.5-flash-lite-preview-06-17")
evaluator_llm = init_chat_model("google_genai:gemini-2.5-flash-lite-preview-06-17", temperature=0.0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Prompts ---
interviewer_prompt = PromptTemplate(
    input_variables=["mode", "company_name", "position", "number_of_questions", "number_of_followup"],
    template="""
        You are an {mode} AI interviewer for a leading tech company called {company_name}, conducting an interview for a {position} position.
        Your goal is to assess the candidate's technical skills, problem-solving abilities, communication skills, and experience relevant to data science roles.
        Maintain a professional yet approachable tone.
        You have access to two tools:
        1. `retrieve_documents`: This tool can search a knowledge base of interview questions related to the {position} position. Use this tool to find relevant questions to ask the candidate.
        2. `retrieve_resume`: This tool can search the candidate's resume to find information about their past projects and experience. Use this tool to ask relevant projects from their resume like {position} projects.
        Start by introducing yourself as the interviewer and asking the candidate to introduce themselves, then ask use tools to retrive a project of you choice in there resume and ask them about it.
        Focus on questions related to the position and the candidate's resume.
        You ask only one Introduction question at the beginning of the interview, then one question about a project from there resume then {number_of_questions} questions about the position from the knowledge base with {number_of_followup} flowup question only if there answer was too vage and incomplete.
        If asked any irrelevant question, respond with: "Sorry, this is out of scope."
        After the interview is finished you output: "Thank you, that's it for today."
        if you use any tool print"tool used: `tool_name`"
        when you bull a question from the knowldebase specify the number of the question, Example:
        `
        Question one: What challenges do LLMs face in deployment?
        Question twe: What defines a Large Language Model (LLM)?
        `
        to elistrate between main questions and follow-up questions.
        Begin the interview now.
        """
)

evaluator_prompt = PromptTemplate(
    input_variables=["num_of_q", "num_of_follow_up", "position"],
    template="""You are an AI evaluator for a job interview. Your task is to evaluate the candidate's responses based\
        on their relevance, clarity, and depth.
        You will receive one Introduction question, one project question, and {num_of_q} technical questions with up to {num_of_follow_up} follow up questions\
        about {position} position.
        Ignore any irrelevant questions or answers.
        You evaluate each response with a score from 1 to 10, where 1 is the lowest and 10 is the highest.
        The context of the interview is as follows:
            Introduction question:
            Project question:
            Technical questions:
        each question could have a follow-up question, if so you should evaluate the main question only and assume the follow up answer is appended to the main answer.
        Usually the main technical question is in the following format:
            Question one: Example question one?
            Question two: Example question two?
        you should evaluate the main question only and assume the follow up answer is appended to the main answer.
        If you don't have enough information to evaluate a Technical question, use the tool `retriever_tool` to get more information about the question.
        You should output the evaluation in the following format:
        Evaluation:
            1. Introduction question: [score] - [reasoning]
            2. Project question: [score] - [reasoning]
            3. Technical question one: [score] - [reasoning]
            4. Technical question two: [score] - [reasoning]
        """
)

report_writer_prompt = PromptTemplate(
    input_variables=["position", "company_name", "interview_transcript", "evaluation_report"],
    template="""You are an AI HR Report Writer. Your task is to synthesize information from a job interview transcript and its evaluation into a concise, professional report for Human Resources at {company_name}.
        The interview was for a **{position}** position.
        Your report should focus on key takeaways relevant to HR's decision-making, including but not limited to:
        -   **Candidate's Overall Suitability:** A brief summary of whether the candidate seems suitable for the role based on their performance.
        -   **Strengths:** Specific areas where the candidate performed well, supported by examples from the transcript if clear.
        -   **Areas for Development/Weaknesses:** Specific areas where the candidate struggled or showed gaps, supported by examples from the transcript if clear.
        -   **Key Technical Skills Demonstrated:** List any core technical skills (e.g., Python, SQL, ML algorithms, data analysis, specific frameworks) explicitly mentioned or clearly demonstrated by the candidate's answers.
        -   **Problem-Solving Approach:** Insights into how the candidate approaches technical problems or challenges (if discernible).
        -   **Communication Skills:** Assessment of clarity, conciseness, and overall effectiveness of their communication during the interview.
        -   **Relevant Experience Highlights:** Any particularly relevant past projects or experiences highlighted by the candidate.
        -   **Recommendations (Optional):** A high-level recommendation (e.g., "Proceed to next round," "Consider for a different role," "Not a good fit at this time").
        You will be provided with:
        1.  **Full Interview Transcript:** The complete conversation between the recruiter and the candidate.
        2.  **Evaluation Report:** A structured evaluation of the candidate's responses provided by an AI evaluator, including scores and reasoning for each question.
        **Instructions for Report Generation:**
        -   **Conciseness:** Be brief and to the point. HR personnel have limited time.
        -   **Professional Tone:** Maintain a neutral, objective, and professional tone throughout the report.
        -   **Evidence-Based:** Support your points with specific references or inferences from the provided transcript and evaluation. Do NOT invent information.
        -   **Structure:** Organize your report with clear headings for each section (e.g., "Candidate Summary," "Strengths," "Areas for Development," etc.).
        -   **Format:** Present the report as a single, well-formatted text block. Do not include any conversational filler.
        ---
        **Interview Transcript:**
        {interview_transcript}
        ---
        **Evaluation Report:**
        {evaluation_report}

        Dont forget to use `save_report_as_pdf` tool to save the report in a pdf.
        """
)

# --- Placeholder for tools, retriever, etc. ---
# These should be initialized in the Streamlit app or a setup function
# retriever_tool = None
# resume_retriever_tool = None

# === BEGIN: Tool Initialization ===
# Placeholder PDF paths (update as needed)
INTERVIEW_QUESTIONS_PDF = "utils/LLM Interview Questions.pdf"
RESUME_PDF = "utils/Mohamed-Mowina-AI-Resume.pdf"

# Load and split documents
loader = PyPDFLoader(INTERVIEW_QUESTIONS_PDF)
pages = loader.load()
resume_loader = PyPDFLoader(RESUME_PDF)
resume = resume_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pages_split = text_splitter.split_documents(pages)
resume_split = text_splitter.split_documents(resume)

# Initialize the vector stores
vectorstore = Chroma.from_documents(
    documents=pages_split, embedding=embeddings, collection_name="LLMs_interview_questions"
)
resume_vectorstore = Chroma.from_documents(
    documents=resume_split, embedding=embeddings, collection_name="resume"
)

retriever = vectorstore.as_retriever()
resume_retriever = resume_vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_questions",
    "Search and return interview questions related to the {position} position from the knowledge base. Use this to get technical questions to ask the candidate.",
)
resume_retriever_tool = create_retriever_tool(
    resume_retriever,
    "retrieve_resume",
    "Search the candidate's resume to find specific projects, skills, and experiences. Use this to ask detailed questions about their past work, projects, and technical skills.",
)
# === END: Tool Initialization ===

@tool
def save_report_as_pdf(report_content: str, filename: str) -> str:
    """
    Saves the provided report content as a PDF file.

    Args:
        report_content (str): The full text content of the HR report.
        filename (str): The desired name for the PDF file (e.g., "HR_Interview_Report_CandidateX.pdf").
                        Do NOT include path, just the filename. The file will be saved in the current directory.

    Returns:
        str: The full path to the saved PDF file if successful, otherwise an error message.
    """
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    
    safe_filename = os.path.basename(filename)

    try:
        doc = SimpleDocTemplate(safe_filename, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # --- MODIFIED PART START ---
        # Use existing styles or create new ones by copying and modifying,
        # rather than trying to add names that already exist.

        # Example: Use 'h1' for main title, 'h2' for section titles, 'Normal' for body text.
        # You can adjust attributes of these copied styles if needed.
        
        # Adjust 'h1' for center alignment (it's usually left by default)
        h1_style = styles['h1']
        h1_style.alignment = TA_CENTER
        h1_style.spaceAfter = 14 # Add some space after the heading

        # Adjust 'h2' for left alignment (it's usually left by default)
        h2_style = styles['h2']
        h2_style.alignment = TA_LEFT
        h2_style.spaceAfter = 8 # Add some space after the subheading

        # Use 'Normal' style for body text
        body_style = styles['Normal']
        body_style.spaceAfter = 6 # Small space after body paragraphs

        # --- MODIFIED PART END ---

        story = []
        
        # Add a title
        story.append(Paragraph("HR Interview Report", h1_style))
        story.append(Spacer(1, 0.2 * letter[1])) # Add some space

        # Split the report content into paragraphs, preserving newlines
        paragraphs = report_content.split('\n\n')
        
        for para_text in paragraphs:
            if para_text.strip(): # Avoid empty paragraphs
                # Check for specific section headings from your prompt's expected output
                if para_text.strip().startswith((
                    'Candidate\'s Overall Suitability:',
                    'Strengths:',
                    'Areas for Development/Weaknesses:',
                    'Key Technical Skills Demonstrated:',
                    'Problem-Solving Approach:',
                    'Communication Skills:',
                    'Relevant Experience Highlights:',
                    'Recommendations:'
                )):
                    story.append(Paragraph(para_text, h2_style)) # Use h2 style for sections
                else:
                    story.append(Paragraph(para_text, body_style)) # Use Normal style for body content
                story.append(Spacer(1, 0.1 * letter[1])) # Add a small space after each paragraph

        doc.build(story)
        return f"Report successfully saved to: {os.path.abspath(safe_filename)}"
    except Exception as e:
        # It's good to print the full traceback for debugging during development
        import traceback
        traceback.print_exc()
        return f"Error saving report as PDF: {e}"

report_writer_tools = [save_report_as_pdf]

def recruiter(state: AgentState) -> AgentState:
    sys_prompt = SystemMessage(content=interviewer_prompt.format(
        mode=state['mode'],
        company_name=state['company_name'],
        position=state['position'],
        number_of_questions=state['num_of_q'],
        number_of_followup=state['num_of_follow_up']
    ))
    all_messages = [sys_prompt] + state["messages"]
    return {"messages": llm.bind_tools([retriever_tool, resume_retriever_tool]).invoke(all_messages)}

def evaluator(state: AgentState) -> AgentState:
    """
    Evaluates the interview conversation and generates an evaluation report.
    """
    # Create a fresh evaluation
    sys_prompt = evaluator_prompt.format(
        num_of_q=state['num_of_q'],
        num_of_follow_up=state['num_of_follow_up'],
        position=state['position']
    )
    sys_message = SystemMessage(content=sys_prompt)
    all_messages = [sys_message]
    
    # Process the conversation messages for evaluation
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            all_messages.append(HumanMessage(content=f"Candidate: {m.content}"))
        elif isinstance(m, AIMessage) and "that's it for today" not in m.content:
            all_messages.append(AIMessage(content=f"AI Recruiter: {m.content}"))
    
    # Generate a new evaluation
    results = evaluator_llm.bind_tools([retriever_tool]).invoke(all_messages)
    
    # Return the evaluation result
    return {"evaluation_result": results.content}

def report_writer(state: AgentState) -> AgentState:
    """ Generates a report based on the interview transcript and evaluation """
    interviewer_transcript = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            interviewer_transcript.append('Candidate: ' + str(m.content))
        elif isinstance(m, AIMessage):
            if 'Evaluation:\n1. Introduction question' not in m.content:
                interviewer_transcript.append('AI Recruiter: ' + str(m.content))
    
    sys_prompt = report_writer_prompt.format(
        position=state['position'],
        company_name=state['company_name'],
        interview_transcript= '\n'.join(interviewer_transcript),
        evaluation_report=state["evaluation_result"]
    )

    sys_message = SystemMessage(content=sys_prompt)
    # Fix: Use a proper HumanMessage instead of a string
    all_messages = [sys_message, HumanMessage(content="Generate the HR report")]
    result = llm.bind_tools(report_writer_tools).invoke(all_messages)
    return {"report": result.content}

def custom_tools_condition(state):
    # Check if there are any messages
    if not state['messages']:
        return "WAIT_FOR_HUMAN"  # No messages yet, wait for human input
    
    # Get the last message
    last_message = state['messages'][-1]
    
    # Check if it's an AI message with tool calls
    if isinstance(last_message, AIMessage) and getattr(last_message, 'tool_calls', None):
        return "tools"  # Routes to your "tools" node
    # Check if it's an AI message with the end conversation marker
    elif isinstance(last_message, AIMessage) and "that's it for today" in last_message.content:
        return "END_CONVERSATION"  # Routes to Evaluator node
    # Otherwise, wait for human input
    else:
        return "WAIT_FOR_HUMAN"  # Routes to END
    
# --- Workflow builder ---
def reset_handler(state, result):
    """
    Special handler for the evaluation result to reset it instead of appending.
    This works around the Annotated[str, add] type annotation.
    """
    new_state = state.copy()
    
    # Check if the evaluation result starts with the reset prefix
    if "evaluation_result" in result and isinstance(result["evaluation_result"], str) and result["evaluation_result"].startswith("__RESET__"):
        # Replace the evaluation result with the new value (without the prefix)
        new_state["evaluation_result"] = result["evaluation_result"].replace("__RESET__", "", 1)
    else:
        # Normal update
        new_state.update(result)
    
    return new_state

def build_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("recruiter", recruiter)
    tool_node = ToolNode([retriever_tool, resume_retriever_tool])
    workflow.add_node("tools", tool_node)
    workflow.add_node("evaluator", evaluator)
    evaluator_tool_node = ToolNode([retriever_tool])
    workflow.add_node("evaluator_tools", evaluator_tool_node)
    workflow.add_node("report_writer", report_writer)
    report_writer_tool_node = ToolNode([save_report_as_pdf])
    workflow.add_node("report_writer_tools", report_writer_tool_node)
    # Add edges as in the notebook (custom_tools_condition, etc.)
    # ...
    workflow.set_entry_point("recruiter")

    # Define conditional edges (this will be refined based on the agent's output)
    # For now, let's assume the agent will either call a tool or finish
    workflow.add_conditional_edges(
        "recruiter", # from node
        custom_tools_condition, # condition function
        {
            "tools": "tools",  # If tools are called, go to the tools node
            "END_CONVERSATION": "evaluator",  # If conversation ends, go to evaluator
            "WAIT_FOR_HUMAN": END  # If waiting for human, go to end
        }
    )

    # Add edge from tools back to the recruiter
    workflow.add_edge("tools", "recruiter")

    # Add edge from evaluator tools to evaluator
    workflow.add_edge("evaluator_tools", "evaluator")

    # Define edges for evaluator node
    workflow.add_conditional_edges(
        "evaluator",
        tools_condition,
        {
            "tools": "evaluator_tools",
            END: "report_writer"
        }  # This will route based on whether tools are called
    )

    # Add edge from evaluator to report writer
    workflow.add_edge("evaluator", "report_writer")

    # Define edges for report writer node
    workflow.add_conditional_edges(
        "report_writer",
        tools_condition,
        {
            "tools": "report_writer_tools",
            END: END
        }
    )

    return workflow.compile()