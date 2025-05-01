import streamlit as st
from agent_graph import create_workflow, process_message
from langchain_core.messages import HumanMessage, AIMessage
import os
import logging
from dotenv import load_dotenv

# Configure logging with more verbosity
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app_debug.log")
    ]
)
logger = logging.getLogger(__name__)

# Enable verbose logging for LangChain and LangGraph
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langgraph").setLevel(logging.DEBUG)
logging.getLogger("langchain_core").setLevel(logging.DEBUG)
logging.getLogger("langchain.agents").setLevel(logging.DEBUG)
logging.getLogger("langchain.chains").setLevel(logging.DEBUG)
logging.getLogger("langchain.tools").setLevel(logging.DEBUG)
logging.getLogger("agent_graph").setLevel(logging.DEBUG)

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

    # Process query using workflow
    try:
        logger.info(f"Starting workflow with prompt: {prompt[:100]}...")

        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            status_placeholder = st.empty()
            status_placeholder.info("Processing your request...")

            try:
                # Get the formatted response
                full_response = process_message(workflow, prompt)
                
                # Display the response with proper formatting
                response_placeholder.markdown(full_response)
                status_placeholder.success("Processing complete!")

            except Exception as stream_error:
                logger.error(f"Error during processing: {str(stream_error)}", exc_info=True)
                status_placeholder.error(f"Error: {str(stream_error)}")
                raise stream_error

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            logger.debug(f"Updated chat history: {st.session_state.messages}")

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        error_message = f"Error: {str(e)}"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})