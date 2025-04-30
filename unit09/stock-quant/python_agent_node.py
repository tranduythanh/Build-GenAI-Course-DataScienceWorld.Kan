from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from tools import (
    StockPriceTool,
    TechnicalIndicatorTool,
    FinancialReportTool,
    CompanyInfoTool,
    MarketIndexTool,
    StockListTool
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
        TechnicalIndicatorTool(),
        FinancialReportTool(),
        CompanyInfoTool(),
        MarketIndexTool(),
        StockListTool()
    ]

    # 🧾 3. Prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """You are an expert Python developer specializing in stock market data analysis.
            You have access to various tools for analyzing stock data, technical indicators,
            financial reports, and market information.

            Follow this format:
            Question -> Thought -> Action -> Action Input -> Observation -> ... -> Final Answer
            Only return the Final Answer when the task is complete.
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")  # This was missing
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
    
    # Get the output from the response
    output = response.get("output", "")
    logger.debug(f"Extracted output: {output}")
    
    # Create AIMessage for the response
    ai_message = AIMessage(content=output)
    logger.debug(f"Created AIMessage: {ai_message}")
    
    # Update the state with the response
    new_state = {
        "messages": [ai_message],
        "chat_history": state.get("chat_history", []) + [message, ai_message]
    }
    logger.debug(f"Returning new state: {new_state}")
    
    return new_state

def create_python_agent_node():
    """Create Python agent node for the workflow"""
    python_agent = create_python_agent()
    return lambda state: python_node(state, python_agent)