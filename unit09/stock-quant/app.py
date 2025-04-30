import streamlit as st
from agent_graph import create_workflow
from langchain_core.messages import HumanMessage
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize workflow
logger.info("Initializing workflow")
workflow = create_workflow()

# Set page config
st.set_page_config(
    page_title="Stock Analysis App",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Analysis App")
st.markdown("""
This app helps you analyze stocks and market data. You can:
- View stock prices
- Calculate technical indicators
- Get company information
- Analyze market index performance
""")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter your query (e.g., 'Show me the stock price of VIC for the last 30 days')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query using workflow.stream
    try:
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "next": None,
            "agent_scratchpad": [],
            "chat_history": []
        }

        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # Stream the response
            for chunk in workflow.stream(
                initial_state,
                {"recursion_limit": 20}
            ):
                if isinstance(chunk, dict) and "messages" in chunk and chunk["messages"]:
                    last_message = chunk["messages"][-1]
                    if hasattr(last_message, "content"):
                        full_response += last_message.content
                        response_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        error_message = f"Error: {str(e)}"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message}) 