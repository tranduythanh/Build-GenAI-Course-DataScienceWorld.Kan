from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import logging
import json
from datetime import datetime
from config import LLM_MODEL

logger = logging.getLogger(__name__)

def create_llm():
    """Create LLM instance"""
    return ChatOpenAI(
        temperature=0,
        model=LLM_MODEL,
        streaming=True,
        verbose=True
    )

def create_step_back_node():
    """Create a step-back node for improving question quality"""
    logger.info("Initializing step-back node")
    llm = create_llm()
    
    def step_back_node(state):
        """Apply step-back technique to analyze and improve the question"""
        try:
            logger.debug("Step Back - State structure: %s", json.dumps({k: str(type(v)) for k, v in state.items()}))
            logger.info("Step Back - Entering step_back_node function")
            
            last_message = state["messages"][-1].content
            logger.info("Step-back node processing message: %s", last_message)
            
            # Apply step-back technique
            step_back_prompt = f"""
            Given the following question about stock market analysis, break it down into fundamental concepts and requirements.
            Question: {last_message}
            
            Consider:
            1. What specific stock or stocks are being analyzed?
            2. What time period is relevant?
            3. What technical indicators or metrics are needed?
            4. What comparisons or relationships are being asked for?
            5. The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
            
            Then, rephrase the question to be more precise and clear for stock market analysis.
            Return only the rephrased question, nothing else.
            """
            
            logger.info("Step Back - Sending prompt to LLM")
            improved_question = llm.invoke(step_back_prompt).content
            logger.info("Step Back - Received response from LLM")
            logger.info("Improved question: %s", improved_question)
            
            # Replace the message with improved version
            state["messages"][-1] = HumanMessage(content=improved_question)
            logger.info("Step Back - Updated state with improved question")
            
            return state
        except Exception as e:
            logger.error("Error in step-back node: %s", str(e), exc_info=True)
            return state
    
    return step_back_node