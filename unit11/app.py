from typing import List, Optional
import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core.tools import BaseTool
from llama_agent import StockQuantAgent
from llama_tools import create_stock_tools

# Load environment variables
load_dotenv()

# Initialize tools using the new function-based approach
tools: List[BaseTool] = create_stock_tools(api_key=os.getenv("VNQUANT_API_KEY"))

# Initialize agent
agent: StockQuantAgent = StockQuantAgent(
    tools=tools,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Streamlit UI
st.title("📈 Vietnamese Stock Analysis Assistant")
st.markdown("### Ask me about Vietnamese stocks!")

# Add sample queries
st.markdown("""
**Sample questions you can ask:**
- "Xin stock data của mã HAG"
- "Xin technical analysis của mã VIC"
- "Get stock price for FPT from 2024-01-01 to 2024-06-30"
- "Calculate technical indicators for MSN"
""")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Hỏi về cổ phiếu Việt Nam (VD: Xin stock data của mã HAG)"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        try:
            response: str = agent.process_query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg: str = f"Error: {str(e)}"
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
    st.header("🛠️ Available Tools")
    st.write("📊 **Get Stock Data**: Lấy dữ liệu giá cổ phiếu Việt Nam")
    st.write("📈 **Technical Analysis**: Phân tích kỹ thuật (SMA, RSI, MACD, Bollinger Bands)")
    
    st.header("💡 Vietnamese Stock Symbols")
    st.write("- HAG (HAGL Agrico)")
    st.write("- VIC (Vingroup)")
    st.write("- FPT (FPT Corporation)")
    st.write("- VNM (Vinamilk)")
    st.write("- MSN (Masan Group)")
    st.write("- VNINDEX (VN-Index)") 