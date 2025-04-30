import streamlit as st
from agent_graph import create_workflow
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
            "chat_history": []  # Initialize chat history as empty list
        }

        # Log the initial state
        logger.debug(f"Initial state: {initial_state}")
        logger.info(f"Starting workflow with prompt: {prompt[:100]}...")

        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # Add a status message
            status_placeholder = st.empty()
            status_placeholder.info("Processing your request...")

            # Stream the response with detailed logging
            logger.info("Starting workflow stream")
            chunk_count = 0

            try:
                for chunk in workflow.stream(
                    initial_state,
                    {"recursion_limit": 20, "configurable": {"debug": True}}
                ):
                    chunk_count += 1
                    logger.debug(f"Received chunk {chunk_count}: {chunk}")
                    logger.debug(f"Chunk type: {type(chunk)}")

                    if isinstance(chunk, dict):
                        logger.debug(f"Chunk keys: {list(chunk.keys())}")
                        
                        # Handle nested message structures
                        messages = None
                        for key, value in chunk.items():
                            if isinstance(value, dict) and "messages" in value:
                                messages = value["messages"]
                                logger.debug(f"Found messages in {key}: {messages}")
                                break
                        
                        if messages:
                            logger.debug(f"Processing messages: {messages}")
                            last_message = messages[-1]
                            logger.debug(f"Last message: {last_message}")
                            logger.debug(f"Last message type: {type(last_message)}")
                            
                            if hasattr(last_message, "content"):
                                logger.debug(f"Message content: {last_message.content}")
                                full_response += last_message.content
                                response_placeholder.markdown(full_response)
                                logger.debug(f"Updated full response: {full_response}")

                                # Update status
                                status_placeholder.success("Processing complete!")
                            else:
                                logger.warning(f"Message has no content attribute: {last_message}")
                        else:
                            logger.warning(f"No messages found in chunk: {chunk}")
                    else:
                        logger.warning(f"Unexpected chunk type: {type(chunk)}")
            except Exception as stream_error:
                logger.error(f"Error during streaming: {str(stream_error)}", exc_info=True)
                status_placeholder.error(f"Stream error: {str(stream_error)}")
                raise stream_error

            logger.info(f"Workflow completed with {chunk_count} chunks")
            logger.debug(f"Final full response: {full_response}")

            if not full_response:
                logger.warning("No response generated from workflow")
                status_placeholder.warning("No response was generated. Please try again.")
                full_response = "I'm sorry, I couldn't process your request. Please try again with a different query."
                response_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            logger.debug(f"Updated chat history: {st.session_state.messages}")

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        error_message = f"Error: {str(e)}"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})