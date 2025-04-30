from typing import Dict, TypedDict, Annotated, Sequence, List, Literal, Union
from langgraph.graph import Graph, StateGraph, END, START, MessagesState
from langgraph.types import Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

import os
import logging
import sys
import uuid
import json
import langgraph

from python_agent_node import create_python_agent_node
from tools import (
    StockPriceTool,
    TechnicalIndicatorTool,
    FinancialReportTool,
    CompanyInfoTool,
    MarketIndexTool,
    StockListTool
)
from agent_types import AgentState, get_next_node
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

def create_translator_node():
    """Create a translator node for language detection and translation"""
    logger.info("Initializing translator node")
    llm = create_llm()
    
    def translator_node(state):
        """Detect language and translate if needed"""
        try:
            # Debug state structure
            logger.debug("Translator - State structure: %s", json.dumps({k: str(type(v)) for k, v in state.items()}))
            
            last_message = state["messages"][-1].content
            logger.info("Translator node processing message: %s", last_message[:100] + "...")
            
            # Detect language
            detect_prompt = f"""
            Determine the language of the following text and respond with ONLY the language code (e.g., 'en' for English, 'vi' for Vietnamese):
            
            Text: {last_message}
            """
            
            language_code = llm.invoke(detect_prompt).content.strip().lower()
            logger.info("Detected language: %s", language_code)
            
            # If not English, translate to English
            if language_code != "en":
                translate_prompt = f"""
                Translate the following text to English. 
                Only return the English translation, nothing else.
                
                Text: {last_message}
                """
                
                translated_text = llm.invoke(translate_prompt).content
                logger.info("Translated text: %s", translated_text[:100] + "...")
                
                # Replace the message with translated version
                state["messages"][-1] = HumanMessage(content=translated_text)
                
                # Store original language for translating back later
                state["original_language"] = language_code
            
            return state
        except Exception as e:
            logger.error("Error in translator node: %s", str(e), exc_info=True)
            return state
    
    return translator_node

def create_step_back_node():
    """Create a step-back node for improving question quality"""
    logger.info("Initializing step-back node")
    llm = create_llm()
    
    def step_back_node(state):
        """Apply step-back technique to analyze and improve the question"""
        try:
            # Debug state structure
            logger.debug("Step Back - State structure: %s", json.dumps({k: str(type(v)) for k, v in state.items()}))
            
            last_message = state["messages"][-1].content
            logger.info("Step-back node processing message: %s", last_message[:100] + "...")
            
            # Apply step-back technique
            step_back_prompt = f"""
            Given the following question about stock market analysis, break it down into fundamental concepts and requirements.
            Question: {last_message}
            
            Consider:
            1. What specific stock or stocks are being analyzed?
            2. What time period is relevant?
            3. What technical indicators or metrics are needed?
            4. What comparisons or relationships are being asked for?
            
            Then, rephrase the question to be more precise and clear for stock market analysis.
            Return only the rephrased question, nothing else.
            """
            
            improved_question = llm.invoke(step_back_prompt).content
            logger.info("Improved question: %s", improved_question[:100] + "...")
            
            # Replace the message with improved version
            state["messages"][-1] = HumanMessage(content=improved_question)
            
            return state
        except Exception as e:
            logger.error("Error in step-back node: %s", str(e), exc_info=True)
            return state
    
    return step_back_node

def create_response_translator_node():
    """Create a response translator node to translate results back to original language"""
    logger.info("Initializing response translator node")
    llm = create_llm()
    
    def response_translator_node(state):
        """Translate response back to original language if needed"""
        try:
            # Debug state structure
            logger.debug("Response Translator - State structure: %s", json.dumps({k: str(type(v)) for k, v in state.items()}))
            
            # Check if we need to translate the response
            if "original_language" in state and state["original_language"] != "en":
                original_language = state["original_language"]
                response = state["messages"][-1].content
                logger.info("Translating response back to %s: %s", original_language, response[:100] + "...")
                
                # Translate response back to original language
                translate_prompt = f"""
                Translate the following English answer to the language with code '{original_language}'. 
                DO NOT translate any stock symbols, technical terms, or numbers.
                Keep all numbers, dates, and technical terms in their original form.
                Only return the translated response, nothing else.
                
                Answer: {response}
                """
                
                translated_response = llm.invoke(translate_prompt).content
                logger.info("Translated response: %s", translated_response[:100] + "...")
                
                # Replace the message with translated version
                state["messages"][-1] = AIMessage(content=translated_response)
                
                # Remove the original language from state
                del state["original_language"]
            
            return state
        except Exception as e:
            logger.error("Error in response translator node: %s", str(e), exc_info=True)
            return state
    
    return response_translator_node

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
    translator_node = create_translator_node()
    step_back_node = create_step_back_node()
    python_agent_node = create_python_agent_node()
    response_translator_node = create_response_translator_node()
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("translator", translator_node)
    workflow.add_node("step_back", step_back_node)
    workflow.add_node("python_agent", python_agent_node)
    workflow.add_node("response_translator", response_translator_node)
    
    # Add edges
    workflow.add_edge("translator", "step_back")
    workflow.add_edge("step_back", "python_agent")
    workflow.add_edge("python_agent", "response_translator")
    workflow.add_edge("response_translator", END)
    
    # Set entry point
    workflow.set_entry_point("translator")
    
    # Compile workflow
    return workflow.compile()

def process_message(workflow, state, message: str):
    """Process a message through the workflow"""
    logger.info("Processing message: %s", message[:100] + "..." if len(message) > 100 else message)
    
    # Create a unique ID for this request for logging
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"Request ID: {request_id}")
    
    # Initialize state correctly
    state = {
        "messages": [HumanMessage(content=message)],
        "next": None  # Let LangGraph determine the next node
    }
    
    # Debug initial state structure
    logger.debug(f"[{request_id}] Initial state keys: {list(state.keys())}")
    logger.debug(f"[{request_id}] Initial state structure: {json.dumps({k: str(type(v)) for k, v in state.items()})}")
    logger.debug(f"[{request_id}] Initial messages length: {len(state['messages'])}")
    
    # Run the workflow with streaming
    logger.info(f"[{request_id}] Starting workflow stream")
    try:
        full_response = ""
        for chunk in workflow.stream(
            state,
            {"recursion_limit": 20, "configurable": {"debug": True}}
        ):
            logger.debug(f"[{request_id}] Received chunk: {chunk}")
            
            if isinstance(chunk, dict):
                # Handle nested message structures
                messages = None
                for key, value in chunk.items():
                    if isinstance(value, dict) and "messages" in value:
                        messages = value["messages"]
                        break
                
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, "content"):
                        full_response += last_message.content
                        logger.debug(f"[{request_id}] Updated full response: {full_response}")
        
        logger.info(f"[{request_id}] Workflow completed")
        logger.debug(f"[{request_id}] Final full response: {full_response}")
        
        if full_response:
            return full_response
        else:
            error_msg = "No response generated from workflow"
            logger.error(f"[{request_id}] {error_msg}")
            return f"Error: {error_msg}"
    except Exception as e:
        logger.error(f"[{request_id}] Error during workflow execution: {str(e)}", exc_info=True)
        return f"Error: {str(e)}" 