# Corrected Interactive Interview with Proper Conversation History
# This fixes the issues in the original notebook

from langgraph.graph import StateGraph, START, END
from typing import Dict, TypedDict, List, Union, Annotated, Sequence
from IPython.display import display, Markdown, Image
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage, BaseMessage, ToolMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages
from langchain_chroma import Chroma
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    '''
    Responsible for the state of the recruiter agent

    args:
        mode: str
        num_of_q: int
        num_of_follow_up: int
        position: str
        company_name: str
        messages: Sequence[BaseMessage]
    '''
    mode: str
    num_of_q: int
    num_of_follow_up: int
    position: str
    company_name: str
    messages: Annotated[list, add_messages]

def setup_interview_system(pdf_path: str, resume_path: str):
    """
    Set up the interview system with knowledge base and resume data
    
    Args:
        pdf_path: Path to the interview questions PDF
        resume_path: Path to the candidate's resume PDF
    
    Returns:
        tuple: (app, llm, embeddings) - compiled workflow, language model, and embeddings
    """
    # Initialize the language model and embeddings
    llm = init_chat_model("google_genai:gemini-2.5-flash-lite-preview-06-17")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load documents
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume file not found: {resume_path}")
    
    pdf_loader = PyPDFLoader(pdf_path)
    resume_loader = PyPDFLoader(resume_path)
    
    try:
        pages = pdf_loader.load()
        resume = resume_loader.load()
        print(f"PDF has been loaded and has {len(pages)} pages")
        print(f"Resume has been loaded and has {len(resume)} pages")
    except Exception as e:
        print(f"Error loading PDFs: {e}")
        raise
    
    # Split pages to chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    pages_split = text_splitter.split_documents(pages)
    resume_split = text_splitter.split_documents(resume)
    print(f"The document has been split into {len(pages_split)} chunks")
    print(f"The resume has been split into {len(resume_split)} chunks")
    
    # Initialize the vector stores
    vectorstore = Chroma.from_documents(
        documents=pages_split, embedding=embeddings, collection_name="LLMs_interview_questions"
    )
    
    resume_vectorstore = Chroma.from_documents(
        documents=resume_split, embedding=embeddings, collection_name="resume"
    )
    
    retriever = vectorstore.as_retriever()
    resume_retriever = resume_vectorstore.as_retriever()
    
    # Create tools
    from langchain.tools.retriever import create_retriever_tool
    
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
    
    tools = [retriever_tool, resume_retriever_tool]
    
    # Create interviewer prompt
    interviewer_prompt = PromptTemplate(
        input_variables=["mode", "company_name", "position", "number_of_questions", "number_of_followup"],
        template="""
You are an {mode} AI interviewer for a leading tech company called {company_name}, conducting an interview for a {position} position.

Your goal is to assess the candidate's technical skills, problem-solving abilities, communication skills, and experience relevant to the {position} role.

Maintain a professional yet approachable tone.

IMPORTANT: You have access to the candidate's resume through the `retrieve_resume` tool. You MUST use this tool to:
1. Find specific projects and experiences from their resume
2. Ask detailed questions about their past work
3. Reference their actual experience when asking questions

You also have access to `retrieve_questions` tool to get relevant interview questions for the {position} position.

INTERVIEW FLOW:
1. Start by introducing yourself and asking the candidate to introduce themselves
2. Use the `retrieve_resume` tool to find a specific project from their resume and ask them about it
3. Ask {number_of_questions} technical questions about the {position} role using the `retrieve_questions` tool
4. Ask up to {number_of_followup} follow-up questions if answers are vague or incomplete

ALWAYS use the tools to access resume content and interview questions. Do not say you don't have access to the resume - you do!

If asked any irrelevant question, respond with: "Sorry, this is out of scope."

After the interview is finished you output: "Thank you, that's it for today."

Begin the interview now.
"""
    )
    
    # Define the recruiter agent function
    def recruiter(state: AgentState) -> AgentState:
        ''' the agent function call llm with a system prompt that customizes the persona of the recruiter '''
        sys_prompt = SystemMessage(content=interviewer_prompt.format(
            mode=state['mode'],
            company_name=state['company_name'],
            position=state['position'],
            number_of_questions=state['num_of_q'],
            number_of_followup=state['num_of_follow_up']
        ))
        
        # Ensure all_messages is a list of BaseMessage objects
        all_messages = [sys_prompt] + state["messages"]
        
        return {"messages": llm.invoke(all_messages)}
    
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Add the agent node
    workflow.add_node("recruiter", recruiter)
    
    # Add tool node
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    
    # Set the entry point
    workflow.set_entry_point("recruiter")
    
    # Define conditional edges
    workflow.add_conditional_edges(
        "recruiter",  # from node
        tools_condition
    )
    
    # Add edge from tools back to the recruiter
    workflow.add_edge("tools", "recruiter")
    
    # Compile the graph
    app = workflow.compile()
    
    return app, llm, embeddings

def run_interview(
    pdf_path: str,
    resume_path: str,
    position: str = "AI Developer",
    company_name: str = "TechCorp",
    mode: str = "friendly",
    num_questions: int = 5,
    num_followup: int = 2,
    max_turns: int = 20
):
    """
    Run the AI interview workflow
    
    Args:
        pdf_path: Path to the interview questions PDF
        resume_path: Path to the candidate's resume PDF
        position: Job position being interviewed for
        company_name: Name of the company conducting the interview
        mode: Interview mode (friendly, professional, etc.)
        num_questions: Number of main questions to ask
        num_followup: Number of follow-up questions allowed
        max_turns: Maximum number of conversation turns
    
    Returns:
        dict: Final state of the interview
    """
    print("Setting up interview system...")
    app, llm, embeddings = setup_interview_system(pdf_path, resume_path)
    
    # Initialize the interview state
    initial_state = {
        "mode": mode,
        "num_of_q": num_questions,
        "num_of_follow_up": num_followup,
        "position": position,
        "company_name": company_name,
        "messages": [HumanMessage(content="Hello, I'm ready for the interview.")]
    }
    
    print(f"\n{'='*60}")
    print(f"Starting interview for {position} position at {company_name}")
    print(f"Mode: {mode}, Questions: {num_questions}, Follow-ups: {num_followup}")
    print(f"{'='*60}\n")
    
    # Run the workflow
    try:
        final_state = app.invoke(initial_state, config={"recursion_limit": max_turns})
        
        print(f"\n{'='*60}")
        print("Interview completed!")
        print(f"{'='*60}")
        
        return final_state
        
    except Exception as e:
        print(f"Error during interview: {e}")
        raise

def run_interactive_interview(
    pdf_path: str,
    resume_path: str,
    position: str = "AI Developer",
    company_name: str = "TechCorp",
    mode: str = "friendly",
    num_questions: int = 5,
    num_followup: int = 2
):
    """
    Run an interactive interview where the user can respond to AI questions
    
    Args:
        pdf_path: Path to the interview questions PDF
        resume_path: Path to the candidate's resume PDF
        position: Job position being interviewed for
        company_name: Name of the company conducting the interview
        mode: Interview mode (friendly, professional, etc.)
        num_questions: Number of main questions to ask
        num_followup: Number of follow-up questions allowed
    """
    print("Setting up interactive interview system...")
    app, llm, embeddings = setup_interview_system(pdf_path, resume_path)
    
    # Initialize the interview state
    state = {
        "mode": mode,
        "num_of_q": num_questions,
        "num_of_follow_up": num_followup,
        "position": position,
        "company_name": company_name,
        "messages": []
    }
    
    print(f"\n{'='*60}")
    print(f"Interactive Interview for {position} position at {company_name}")
    print(f"Mode: {mode}, Questions: {num_questions}, Follow-ups: {num_followup}")
    print("Type 'quit' to end the interview")
    print(f"{'='*60}\n")
    
    turn_count = 0
    max_turns = 50
    
    try:
        while turn_count < max_turns:
            # Run the workflow for one step
            result = app.invoke(state, config={"recursion_limit": 1})
            
            # Get the latest AI message
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                latest_ai_message = ai_messages[-1]
                print(f"\n🤖 Interviewer: {latest_ai_message.content}")
                
                # Check if interview is finished
                if "Thank you, that's it for today" in latest_ai_message.content:
                    print("\n🎉 Interview completed!")
                    break
            
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'end']:
                print("\n👋 Interview ended by user.")
                break
            
            # Add user message to state
            state["messages"].append(HumanMessage(content=user_input))
            turn_count += 1
            
    except KeyboardInterrupt:
        print("\n\n👋 Interview interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during interview: {e}")
        raise
    
    return state

def test_tools(pdf_path: str, resume_path: str):
    """
    Test the retrieval tools to ensure they're working properly
    
    Args:
        pdf_path: Path to the interview questions PDF
        resume_path: Path to the candidate's resume PDF
    """
    print("Testing retrieval tools...")
    
    try:
        app, llm, embeddings = setup_interview_system(pdf_path, resume_path)
        
        # Test resume retrieval
        print("\n" + "="*50)
        print("TESTING RESUME RETRIEVAL")
        print("="*50)
        
        # Create a simple test state
        test_state = {
            "mode": "friendly",
            "num_of_q": 3,
            "num_of_follow_up": 1,
            "position": "AI Developer",
            "company_name": "TestCorp",
            "messages": [HumanMessage(content="Tell me about my projects")]
        }
        
        # Run one step to see if tools are called
        result = app.invoke(test_state, config={"recursion_limit": 1})
        
        # Check if tools were used
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        
        if tool_messages:
            print("✅ Tools are being called successfully!")
            for tool_msg in tool_messages:
                print(f"\n🔧 Tool: {tool_msg.tool_name}")
                print(f"Result: {tool_msg.content[:300]}...")
        else:
            print("❌ No tools were called")
            
        if ai_messages:
            print(f"\n🤖 AI Response: {ai_messages[-1].content}")
            
    except Exception as e:
        print(f"❌ Error testing tools: {e}")
        raise

def print_interview_summary(final_state: dict):
    """
    Print a summary of the interview conversation
    
    Args:
        final_state: The final state from the interview workflow
    """
    print(f"\n{'='*60}")
    print("INTERVIEW SUMMARY")
    print(f"{'='*60}")
    
    messages = final_state.get("messages", [])
    
    for i, message in enumerate(messages, 1):
        if isinstance(message, HumanMessage):
            print(f"\n👤 Turn {i} - Candidate:")
            print(f"   {message.content}")
        elif isinstance(message, AIMessage):
            print(f"\n🤖 Turn {i} - Interviewer:")
            print(f"   {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"\n🔧 Turn {i} - Tool Used:")
            print(f"   Tool: {message.tool_name}")
            print(f"   Result: {message.content[:200]}...")
    
    print(f"\n{'='*60}")
    print(f"Total turns: {len(messages)}")
    print(f"Interview mode: {final_state.get('mode', 'Unknown')}")
    print(f"Position: {final_state.get('position', 'Unknown')}")
    print(f"Company: {final_state.get('company_name', 'Unknown')}")
    print(f"{'='*60}")

# Example usage functions
def example_automated_interview():
    """Example of running an automated interview"""
    try:
        final_state = run_interview(
            pdf_path="LLM Interview Questions.pdf",
            resume_path="Mohamed-Mowina-AI-Resume.pdf",
            position="AI Developer",
            company_name="OpenAI",
            mode="friendly",
            num_questions=3,
            num_followup=1
        )
        
        print_interview_summary(final_state)
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please ensure the PDF files are in the correct location.")
    except Exception as e:
        print(f"Error: {e}")

def example_interactive_interview():
    """Example of running an interactive interview"""
    try:
        final_state = run_interactive_interview(
            pdf_path="LLM Interview Questions.pdf",
            resume_path="Mohamed-Mowina-AI-Resume.pdf",
            position="AI Developer",
            company_name="OpenAI",
            mode="friendly",
            num_questions=3,
            num_followup=1
        )
        
        print_interview_summary(final_state)
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please ensure the PDF files are in the correct location.")
    except Exception as e:
        print(f"Error: {e}")

def example_test_tools():
    """Example of testing the retrieval tools"""
    try:
        test_tools(
            pdf_path="LLM Interview Questions.pdf",
            resume_path="Mohamed-Mowina-AI-Resume.pdf"
        )
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please ensure the PDF files are in the correct location.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # You can run either example here
    print("Choose interview mode:")
    print("1. Automated interview")
    print("2. Interactive interview")
    print("3. Test tools (debug mode)")
    
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        example_automated_interview()
    elif choice == "2":
        example_interactive_interview()
    elif choice == "3":
        example_test_tools()
    else:
        print("Invalid choice. Running automated interview as default.")
        example_automated_interview() 