from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

import logging
import sys
import json
from node_planner import create_planner_node
from node_python_agent import create_python_agent_node
from node_step_back import create_step_back_node
from node_summary import create_summary_node
from agent_types import AgentState
from config import LLM_MODEL

# Configure logging with more verbosity
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_llm():
    """Create LLM instance"""
    return ChatOpenAI(
        temperature=0,
        model=LLM_MODEL,
        streaming=True,
        verbose=True
    )

def create_agent_prompt(agent_type: str) -> ChatPromptTemplate:
    system_message = (
        "You are a helpful AI assistant specialized in stock market analysis. "
        "Use the provided tools to analyze stock data and provide insights. "
        "Execute what you can to make progress."
        f"\nYou can only do {agent_type}."
        "\nIMPORTANT: You MUST include FINAL ANSWER in your response when you have completed your specific task:"
        "\n- As Python agent: Include FINAL ANSWER after you have successfully performed the analysis"
        "\nDo not include FINAL ANSWER until you have actually completed your specific task."
    )
    
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ])

def create_agent_executor(llm, tools, agent_type: str):
    """Creates a ReAct style agent manually"""
    prompt = create_agent_prompt(agent_type)
    
    def run_agent(state):
        # Extract messages from state
        messages = state.get("messages", [])
        
        # Format messages for the agent prompt
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append(("human", msg.content))
            elif isinstance(msg, AIMessage):
                formatted_messages.append(("ai", msg.content))
            elif isinstance(msg, SystemMessage):
                formatted_messages.append(("system", msg.content))
        
        # Prepare the input for the LLM
        input_data = {
            "messages": messages,
            "input": f"You have access to the following tools: {[t.name for t in tools]}. " +
                    f"Use them to respond to the user's request."
        }
        
        # Get response from LLM
        response = llm.invoke(prompt.format_prompt(**input_data).to_messages())
        
        return {
            "messages": messages + [response]
        }
    
    return run_agent


def router(state: AgentState) -> str:
    """Route to the Python agent"""
    logger.debug("Router - State structure: %s", json.dumps({k: str(type(v)) for k, v in state.items()}))
    logger.debug("Router - State keys: %s", list(state.keys()))
    
    messages = state["messages"]
    last_message = messages[-1]
    logger.info("Router processing message: %s", last_message.content[:100] + "..." if len(last_message.content) > 100 else last_message.content)
    
    # Always route to Python agent
    return "python_agent"

def create_workflow():
    """Create the agent workflow"""
    logger.info("Creating agent workflow")
    
    # Create nodes
    logger.info("Creating step-back node")
    step_back_node = create_step_back_node()
    logger.info("Creating planner node")
    planner_node = create_planner_node()
    logger.info("Creating python agent node")
    python_agent_node = create_python_agent_node()
    logger.info("Creating summary node")
    summary_node = create_summary_node()
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    logger.info("Adding nodes to workflow")
    workflow.add_node("step_back", step_back_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("python_agent", python_agent_node)
    workflow.add_node("summary", summary_node)
    
    # Add edges
    logger.info("Adding edges to workflow")
    workflow.add_edge("step_back", "planner")
    workflow.add_edge("planner", "python_agent")
    workflow.add_edge("python_agent", "summary")
    workflow.add_edge("summary", END)
    
    # Set entry point
    logger.info("Setting entry point to step_back")
    workflow.set_entry_point("step_back")
    
    # Compile workflow
    logger.info("Compiling workflow")
    return workflow.compile()

def process_message(workflow, message: str):
    """Process a message through the workflow and return the response"""
    logger.info("Processing message: %s", message)
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "next": None,
        "chat_history": []
    }
    
    # Format the response to include intermediate steps
    formatted_response = []
    final_answer = None
    intermediate_steps = []
    
    # Run the workflow with streaming
    for chunk in workflow.stream(
        initial_state,
        {"recursion_limit": 20, "configurable": {"debug": True}}
    ):
        logger.debug(f"Received chunk: {chunk}")
        
        if isinstance(chunk, dict):
            # Handle nested message structures
            for key, value in chunk.items():
                if isinstance(value, dict) and "messages" in value:
                    messages = value["messages"]
                    for msg in messages:
                        if hasattr(msg, "content"):
                            content = msg.content
                            # Check if this is a final answer
                            if isinstance(msg, AIMessage) and ("FINAL ANSWER" in content or "Final Answer ->" in content):
                                final_answer = content
                            # Check if this is an intermediate step
                            elif isinstance(msg, HumanMessage) and "Action:" in content:
                                intermediate_steps.append(f"\n### {content}")
                            elif isinstance(msg, AIMessage) and "Observation:" in content:
                                intermediate_steps.append(f"{content}")
    
    print(f'Final answer: {final_answer}')
    print(f'Intermediate steps: {intermediate_steps}')
    
    # Compose the final response
    if final_answer:
        formatted_response.append("## Final Answer")
        formatted_response.append(final_answer)
    
    if intermediate_steps:
        formatted_response.append("\n## Analysis Process")
        formatted_response.extend(intermediate_steps)
    
    # If no response was generated, return a default message
    if not formatted_response:
        return "I'm sorry, I couldn't process your request. Please try again with a different query."
    
    return "\n".join(formatted_response) 