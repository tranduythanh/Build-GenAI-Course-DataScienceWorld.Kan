from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from tools import (
    StockPriceTool,
    TechnicalIndicatorTool
)
from config import LLM_MODEL

import logging
logger = logging.getLogger(__name__)


def create_python_agent():
    logger.info("Initializing Python agent")

    # 🧠 1. Initialize the LLM
    llm = ChatOpenAI(
        temperature=0,
        model=LLM_MODEL,
        streaming=True,
        verbose=True
    )

    # 🧰 2. List of tools the agent can use
    tools = [
        StockPriceTool(),
        TechnicalIndicatorTool()
    ]

    # 🧾 3. Prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """You are an expert Python developer specializing in stock market data analysis.
            You have access to various tools for analyzing stock data and technical indicators.

            Follow this format:
            Question -> Plan -> Thought -> Action -> Action Input -> Observation -> ... -> Final Answer
            
            Planning Steps:
            1. Analyze the question and break it down into key components
            2. Identify which tools and data sources will be needed
            3. Determine the sequence of operations required
            4. Consider any potential challenges or edge cases
            
            Only return the Final Answer when the task is complete.
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # 🤖 4. Create the core agent
    agent = create_openai_functions_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # 🧠 5. Create agent executor
    executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    # 🧩 6. Wrap executor in RunnableWithMessageHistory
    return RunnableWithMessageHistory(
        executor,
        get_session_history=lambda session_id: InMemoryChatMessageHistory(),
        input_messages_key="input",
        history_messages_key="chat_history"
    )


def python_node(state, agent):
    """Process the state through the Python agent"""
    logger.debug(f"Entering python_node with state: {state}")
    
    # Get the current message from the state
    message = state.get("messages", [])[-1]
    logger.debug(f"Current message: {message}")
    
    # Run the agent with the message content and session_id
    logger.debug("Invoking agent with input and chat history")
    response = agent.invoke(
        {
            "input": message.content,
            "chat_history": state.get("chat_history", [])
        },
        {
            "configurable": {
                "session_id": "default_session"  # Using a default session ID
            }
        }
    )
    logger.debug(f"Agent response: {response}")
    
    # Get the output and intermediate steps from the response
    output = response.get("output", "")
    intermediate_steps = response.get("intermediate_steps", [])
    logger.debug(f"Extracted output: {output}")
    logger.debug(f"Intermediate steps: {intermediate_steps}")
    
    # Create messages for intermediate steps
    intermediate_messages = []
    for step in intermediate_steps:
        action, observation = step
        # Add the tool's action to chat history
        intermediate_messages.append(HumanMessage(content=f"Action: {action.tool}\nAction Input: {action.tool_input}"))
        # Add the tool's observation to chat history
        intermediate_messages.append(AIMessage(content=f"Observation: {observation}"))
    
    # Create AIMessage for the final response
    ai_message = AIMessage(content=output)
    logger.debug(f"Created AIMessage: {ai_message}")
    
    # Update the state with all messages including intermediate steps
    new_state = {
        "messages": [ai_message],
        "chat_history": state.get("chat_history", []) + [message] + intermediate_messages + [ai_message],
        "intermediate_steps": intermediate_steps  # 👈 giữ lại raw steps
    }
    logger.debug(f"Returning new state: {new_state}")
    
    return new_state

def create_python_agent_node():
    """Create Python agent node for the workflow"""
    python_agent = create_python_agent()
    return lambda state: python_node(state, python_agent)