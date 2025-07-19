# src/dynamic_workflow.py

# Standard library imports
import os

# Type annotations
from typing import Dict, Any, List, TypedDict, Annotated, Sequence
from operator import add

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain.tools.retriever import create_retriever_tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF generation
from src.pdf_utils import generate_pdf

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
    resume_path: str | None  # Path to the candidate's resume
    questions_path: str | None  # Path to the interview questions PDF

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
        After the interview is finished you output this sentance exacly: "Thank you, that's it for today."
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
        """
)

# --- Vector Store and Retriever Setup ---
# Default files
INTERVIEW_QUESTIONS_PDF = "data/default_questions.pdf"
DEFAULT_RESUME_PDF = "data/default_resume.pdf"

# Text splitter for document processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def initialize_questions_retriever(questions_path=None):
    """
    Initialize the questions retriever with the provided questions PDF path or use the default.
    
    Args:
        questions_path (str, optional): Path to the questions PDF. Defaults to None.
    
    Returns:
        The questions retriever tool.
    """
    # Use the provided questions path or fall back to the default
    questions_file = questions_path if questions_path and os.path.exists(questions_path) else INTERVIEW_QUESTIONS_PDF
    
    # Generate a unique ID for this session to avoid caching issues
    import uuid
    import time
    session_id = f"{uuid.uuid4().hex}_{int(time.time())}"
    collection_name = f"questions_{session_id}"
    
    print(f"Loading interview questions from: {questions_file}")
    
    # Always create a fresh collection for each request
    # Load and split the questions
    loader = PyPDFLoader(questions_file)
    pages = loader.load()
    
    # Print the first few lines of the questions for debugging
    print(f"Questions content preview: {pages[0].page_content[:100]}...")
    
    pages_split = text_splitter.split_documents(pages)
    
    # Initialize the questions vector store with a unique collection name
    questions_vectorstore = Chroma.from_documents(
        documents=pages_split, 
        embedding=embeddings, 
        collection_name=collection_name
    )
    
    # Create the retriever with search kwargs for better results
    questions_retriever = questions_vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Return top 3 chunks for better context
    )
    
    # Create the questions retriever tool
    questions_retriever_tool = create_retriever_tool(
        questions_retriever,
        "retrieve_questions",
        "Search and return interview questions related to the {position} position from the knowledge base. Use this to get technical questions to ask the candidate.",
    )
    
    return questions_retriever_tool

def initialize_resume_retriever(resume_path=None):
    """
    Initialize the resume retriever with the provided resume path or use the default.
    
    Args:
        resume_path (str, optional): Path to the resume PDF. Defaults to None.
    
    Returns:
        The resume retriever tool.
    """
    # Use the provided resume path or fall back to the default
    resume_file = resume_path if resume_path and os.path.exists(resume_path) else DEFAULT_RESUME_PDF
    
    # Generate a unique ID for this session to avoid caching issues
    import uuid
    import time
    session_id = f"{uuid.uuid4().hex}_{int(time.time())}"
    collection_name = f"resume_{session_id}"
    
    print(f"Loading resume from: {resume_file}")
    
    # Always create a fresh collection for each request
    # Load and split the resume
    resume_loader = PyPDFLoader(resume_file)
    resume = resume_loader.load()
    
    # Print the first few lines of the resume for debugging
    print(f"Resume content preview: {resume[0].page_content[:100]}...")
    
    resume_split = text_splitter.split_documents(resume)
    
    # Initialize the resume vector store with a unique collection name
    resume_vectorstore = Chroma.from_documents(
        documents=resume_split, 
        embedding=embeddings, 
        collection_name=collection_name
    )
    
    # Create the retriever with search kwargs for better results
    resume_retriever = resume_vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Return top 3 chunks for better context
    )
    
    # Create the resume retriever tool
    resume_retriever_tool = create_retriever_tool(
        resume_retriever,
        "retrieve_resume",
        "Search the candidate's resume to find specific projects, skills, and experiences. Use this to ask detailed questions about their past work, projects, and technical skills.",
    )
    
    return resume_retriever_tool

def pdf_generator_node(state: AgentState) -> AgentState:
    """Generate PDF from the report content"""
    if not state.get("report"):
        return {"pdf_path": None}
    
    # Generate a meaningful filename
    filename = f"HR_Report_{state['company_name']}_{state['position']}.pdf".replace(" ", "_")
    
    try:
        # Call the generate_pdf function directly
        pdf_path = generate_pdf(state["report"], filename=filename)
        return {"pdf_path": pdf_path}
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return {"pdf_path": None}

def recruiter(state: AgentState) -> AgentState:
    # Initialize both retrievers with the paths from the state
    resume_retriever_tool = initialize_resume_retriever(state.get("resume_path"))
    questions_retriever_tool = initialize_questions_retriever(state.get("questions_path"))
    
    # Check if the last message is a tool call for retrieve_resume or retrieve_questions
    if state["messages"] and isinstance(state["messages"][-1], AIMessage) and getattr(state["messages"][-1], "tool_calls", None):
        for tool_call in state["messages"][-1].tool_calls:
            if tool_call.get("name") == "retrieve_resume":
                # Handle the resume retrieval directly
                query = tool_call.get("args", {}).get("query", "")
                print(f"Direct resume query: {query}")
                
                # Execute the tool manually
                result = resume_retriever_tool.invoke({"query": query})
                print(f"Resume retrieval result: {result[:200]}...")
                
                # Create a tool message with the result
                from langchain_core.messages import ToolMessage
                tool_message = ToolMessage(
                    content=result,
                    tool_call_id=tool_call.get("id", ""),
                    name="retrieve_resume"
                )
                
                # Return the tool message
                return {"messages": tool_message}
            
            elif tool_call.get("name") == "retrieve_questions":
                # Handle the questions retrieval directly
                query = tool_call.get("args", {}).get("query", "")
                print(f"Direct questions query: {query}")
                
                # Execute the tool manually
                result = questions_retriever_tool.invoke({"query": query})
                print(f"Questions retrieval result: {result[:200]}...")
                
                # Create a tool message with the result
                from langchain_core.messages import ToolMessage
                tool_message = ToolMessage(
                    content=result,
                    tool_call_id=tool_call.get("id", ""),
                    name="retrieve_questions"
                )
                
                # Return the tool message
                return {"messages": tool_message}
    
    # Normal flow for non-tool messages
    sys_prompt = SystemMessage(content=interviewer_prompt.format(
        mode=state['mode'],
        company_name=state['company_name'],
        position=state['position'],
        number_of_questions=state['num_of_q'],
        number_of_followup=state['num_of_follow_up']
    ))
    all_messages = [sys_prompt] + state["messages"]
    return {"messages": llm.bind_tools([questions_retriever_tool, resume_retriever_tool]).invoke(all_messages)}

def evaluator(state: AgentState) -> AgentState:
    """
    Evaluates the interview conversation and generates an evaluation report.
    """
    # Initialize the questions retriever with the path from the state
    questions_retriever_tool = initialize_questions_retriever(state.get("questions_path"))
    
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
    results = evaluator_llm.bind_tools([questions_retriever_tool]).invoke(all_messages)
    
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
    all_messages = [sys_message, HumanMessage(content="Generate the HR report")]
    result = llm.invoke(all_messages)
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

# Create a dynamic tool node that uses the current paths
def create_dynamic_tool_node(state):
    """Create a tool node with tools based on the current state"""
    # Initialize both retrievers with the current paths
    resume_retriever_tool = initialize_resume_retriever(state.get("resume_path"))
    questions_retriever_tool = initialize_questions_retriever(state.get("questions_path"))
    # Return a tool node with both tools
    return ToolNode([questions_retriever_tool, resume_retriever_tool])

def build_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("recruiter", recruiter)
    
    # Add a dynamic tool node that will be created for each request
    workflow.add_node("tools", create_dynamic_tool_node)
    
    workflow.add_node("evaluator", evaluator)
    
    # Create a dynamic evaluator tool node
    def create_evaluator_tool_node(state):
        questions_retriever_tool = initialize_questions_retriever(state.get("questions_path"))
        return ToolNode([questions_retriever_tool])
    
    workflow.add_node("evaluator_tools", create_evaluator_tool_node)
    workflow.add_node("report_writer", report_writer)
    workflow.add_node("pdf_generator", pdf_generator_node)
    workflow.set_entry_point("recruiter")

    # Define conditional edges
    workflow.add_conditional_edges(
        "recruiter",
        custom_tools_condition,
        {
            "tools": "tools",
            "END_CONVERSATION": "evaluator",
            "WAIT_FOR_HUMAN": END
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
        }
    )

    # Add edge from evaluator to report writer
    workflow.add_edge("evaluator", "report_writer")

    # Add edge from report writer to pdf generator
    workflow.add_edge("report_writer", "pdf_generator")
    
    # Add edge from pdf generator to END
    workflow.add_edge("pdf_generator", END)

    return workflow.compile()