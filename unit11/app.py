import streamlit as st
import os
from dotenv import load_dotenv
from llama_agent import StockQuantAgent
from llama_tools import create_stock_tools

# Load environment variables
load_dotenv()

# Initialize tools using the new function-based approach
tools = create_stock_tools(api_key=os.getenv("VNQUANT_API_KEY"))

# Initialize agent
agent = StockQuantAgent(
    tools=tools,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Streamlit UI
st.title("Stock Quant Analysis")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about the stock market?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        try:
            response = agent.process_query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        agent.clear_memory()
        st.rerun()
    
    # Display chat history
    st.header("Chat History")
    for message in st.session_state.messages:
        st.text(f"{message['role']}: {message['content']}")

# Information about available tools
with st.sidebar:
    st.header("Available Tools")
    st.write("- Get stock price data")
    st.write("- Calculate technical indicators")
    st.write("- Collect web data")
    st.write("- Generate finance reports") 