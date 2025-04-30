import getpass
import os
from typing import Annotated, List, Dict, Any, Tuple, Literal

# Import directly from pydantic v1 for compatibility
from pydantic.v1 import BaseModel, SecretStr

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool, BaseTool
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command

# Set up environment variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError(
        "Please set the TAVILY_API_KEY environment variable. "
        "You can get an API key from https://tavily.com/"
    )

# Set up tools
tavily_tool = TavilySearchResults(max_results=5, tavily_api_key=TAVILY_API_KEY)

# Python REPL tool
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str


# Create prompt templates for agents
def create_agent_prompt(agent_type: str) -> ChatPromptTemplate:
    system_message = (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        f"\nYou can only do {agent_type}. You are working with colleagues."
        "\nIMPORTANT: You MUST include FINAL ANSWER in your response when you have completed your specific task:"
        "\n- As researcher: Include FINAL ANSWER after you have successfully retrieved the UK's GDP data for the past 5 years"
        "\n- As chart generator: Include FINAL ANSWER after you have successfully created and displayed the line chart"
        "\nDo not include FINAL ANSWER until you have actually completed your specific task."
    )
    
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ])


# Set up the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# Create the ReAct-style agent functionality without using create_react_agent
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
        
        # Create a structured response similar to what create_react_agent would return
        return {
            "messages": messages + [response]
        }
    
    return run_agent


def get_next_node(last_message: BaseMessage, goto: str):
    if isinstance(last_message, AIMessage) and "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto


# Create research agent
research_executor = create_agent_executor(
    llm,
    [tavily_tool],
    "research"
)

def research_node(
    state: MessagesState,
) -> Command[Literal["chart_generator", END]]:
    result = research_executor(state)
    last_message = result["messages"][-1]
    goto = get_next_node(last_message, "chart_generator")
    
    # Wrap in a human message if it's an AI message
    if isinstance(last_message, AIMessage):
        wrapped_message = HumanMessage(
            content=last_message.content, name="researcher"
        )
    else:
        wrapped_message = last_message
    
    return Command(
        update={
            "messages": state["messages"] + [wrapped_message],
        },
        goto=goto,
    )


# Create chart generator agent
chart_executor = create_agent_executor(
    llm,
    [python_repl_tool],
    "chart generation"
)

def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    result = chart_executor(state)
    last_message = result["messages"][-1]
    goto = get_next_node(last_message, "researcher")
    
    # Wrap in a human message if it's an AI message
    if isinstance(last_message, AIMessage):
        wrapped_message = HumanMessage(
            content=last_message.content, name="chart_generator"
        )
    else:
        wrapped_message = last_message
    
    return Command(
        update={
            "messages": state["messages"] + [wrapped_message],
        },
        goto=goto,
    )


# Create the workflow
workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("chart_generator", chart_node)

workflow.add_edge(START, "researcher")
graph = workflow.compile()


if __name__ == "__main__":
    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="First, get the UK's GDP over the past 5 years, then make a line chart of it. "
                    "Once you make the chart, finish.",
                )
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 10},
    )
    for s in events:
        print(s)
        print("----")