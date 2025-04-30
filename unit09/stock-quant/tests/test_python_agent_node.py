import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from python_agent_node import python_node, create_python_agent
from agent_types import AgentState


@pytest.fixture
def dummy_state():
    return {
        "messages": [
            HumanMessage(content="What is the stock price of VIC (a Vietnamese stock) today? Please analyze its recent performance.")
        ]
    }

@pytest.fixture
def python_agent():
    return create_python_agent()

def test_python_agent_basic(dummy_state, python_agent):
    result: Command = python_node(dummy_state, python_agent)
    assert isinstance(result, Command)
    assert "messages" in result.update
    assert isinstance(result.update["messages"][-1], HumanMessage)
    assert result.goto in ["sql_agent", "END"]
    print("\nAgent response:", result.update["messages"][-1].content)

def test_python_agent_final_answer(dummy_state, python_agent):
    # Force message to contain FINAL ANSWER
    response_msg = AIMessage(content="FINAL ANSWER: The stock is up.")
    dummy_state["messages"].append(response_msg)
    
    result: Command = python_node(dummy_state, python_agent)
    assert result.goto == "END"

def test_python_agent_skip_if_already_processed(python_agent):
    state = {
        "messages": [
            HumanMessage(content="Already handled", name="python_agent")
        ]
    }
    result: Command = python_node(state, python_agent)
    assert result.goto == "sql_agent"
    assert result.update["messages"] == state["messages"]
