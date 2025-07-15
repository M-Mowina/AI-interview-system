from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage, BaseMessage, ToolMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages
from langchain_chroma import Chroma
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


# intialize the llm
llm = init_chat_model("google_genai:gemini-2.5-flash-lite-preview-06-17")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# test the llm
#print("LLM test: ", llm.invoke(input=[HumanMessage(content="Hello")]))

# test the embeddings
#print("Embeddings test: ", len(embeddings.embed_query("Hello")))

# Define the state of the agent
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


# Load a decument
pdf_path = r"C:\Users\mohamed mowina\Desktop\Projects\Talent-talk\utils\LLM Interview Questions.pdf"
resume_path = r"C:\Users\mohamed mowina\Desktop\Projects\Talent-talk\utils\Mohamed-Mowina-AI-Resume.pdf"

# Safety measure I have put for debugging purposes :)
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path) # This loads the PDF
resume_loader = PyPDFLoader(resume_path) # This loads the PDF

# Checks if the PDF is there
try:
    pages = pdf_loader.load()
    resume = resume_loader.load()
    #print(f"PDF has been loaded and has {len(pages)} pages")
    #print(f"Resume has been loaded and has {len(resume)} pages")
except Exception as e:
    #print(f"Error loading PDFs: {e}")
    raise

# Split pages to chunks
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

pages_split = text_spliter.split_documents(pages)
resume_split = text_spliter.split_documents(resume)
#print(f"The document has been split into {len(pages_split)} chunks")
#print(f"The resume has been split into {len(resume_split)} chunks")

# Intialize the vector store
vectorstore = Chroma.from_documents(
    documents=pages_split, embedding=embeddings, collection_name="LLMs_interview_questions", persist_directory="/content/LLMs_interview_questions"
)

resume_vectorstore = Chroma.from_documents(
    documents=resume_split, embedding=embeddings, collection_name="resume", persist_directory="/content/resume"
)

retriever = vectorstore.as_retriever()
resume_retriever = resume_vectorstore.as_retriever()


# Define the tools
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_questions",
    "Search and return question related to the position from the knoldgebase.",
)

resume_retriever_tool = create_retriever_tool(
    resume_retriever,
    "retrieve_resume",
    "Search resume and return related projects done by the candidate that is rlated to the position.",
)

tools = [retriever_tool, resume_retriever_tool]

# test the tools
#print("Tools test: ", resume_retriever_tool.invoke({"query": "SIEM"}))


# Define the system prompt
interviewer_prompt = PromptTemplate(
    input_variables=["mode", "company_name", "position", "number_of_questions", "number_of_followup"],
    template="""
You are an {mode} AI interviewer for a leading tech company called {company_name}, conducting an interview for a {position} position.

Your goal is to assess the candidate's technical skills, problem-solving abilities, communication skills, and experience relevant to data science roles.

Maintain a professional yet approachable tone.

You have access to two tools:
1. `retrieve_documents`: This tool can search a knowledge base of interview questions related to the {position} position. Use this tool to find relevant questions to ask the candidate.
2. `retrieve_resume`: This tool can search the candidate's resume to find information about their past projects and experience. Use this tool to ask relevant questions about their background.

Start by introducing yourself as the interviewer and asking the candidate to introduce themselves, then ask them about a project of you choice in there resume.

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

def recruiter(state: AgentState) -> AgentState:
  ''' the agent function call llm with a system prompt that costimize the persona of the recruiter '''
  sys_prompt = SystemMessage(content=interviewer_prompt.format(
      mode = state['mode'],
      company_name = state['company_name'],
      position = state['position'],
      number_of_questions = state['num_of_q'],
      number_of_followup = state['num_of_follow_up'] # Corrected key name to match AgentState
  ))

  # Ensure all_messages is a list of BaseMessage objects
  # The input to invoke should be a list of BaseMessage
  all_messages = [sys_prompt] + state["messages"]

  return {"messages": llm.bind_tools(tools).invoke(all_messages)}

# test the agent
init_state = AgentState(
    mode = "friendly",
    num_of_q = 5,
    num_of_follow_up = 2,
    position = "AI developer",
    company_name = "OpenAI",
    messages = [HumanMessage(content="Hi")]
)


#recruiter(init_state)["messages"].pretty_print()

# Define the graph
workflow = StateGraph(AgentState)

# Add the agent node
workflow.add_node("recruiter", recruiter)

# Add tool node
tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)

# Set the entry point
workflow.set_entry_point("recruiter")

# Define conditional edges (this will be refined based on the agent's output)
# For now, let's assume the agent will either call a tool or finish
workflow.add_conditional_edges(
    "recruiter", # from node
    tools_condition
)

# Add edge from tools back to the recruiter
workflow.add_edge("tools", "recruiter")

# Compile the graph
app = workflow.compile()

# Run the chat loop
def chat_loop(initial_state: AgentState):
  """
  Runs a conversational loop with the LangGraph recruiter workflow.

  Args:
    initial_state: The initial state dictionary for the workflow.

  Returns:
    The final state of the workflow after the loop ends.
  """
  current_state = initial_state
  while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
      break

    # Add user message to the state
    current_state["messages"].append(HumanMessage(content=user_input))

    # Run the workflow
    result = app.invoke(current_state)

    # Update the current state with the result
    current_state = result

    # Print the AI's response
    ai_message = current_state["messages"][-1]
    if isinstance(ai_message, AIMessage):
      print(f"AI Recruiter:\n")
      ai_message.pretty_print()
    elif isinstance(ai_message, ToolMessage):
      print(f"AI Recruiter (Tool Output):\n")
      ai_message.pretty_print()
    else:
      print(f"AI Recruiter (Other Message Type): \n")
      ai_message.pretty_print()

  return current_state

# Start the chat loop with the initial state
# final_interview_state = chat_loop(init_state)

# run the app
if __name__ == "__main__":
  state = app.run()