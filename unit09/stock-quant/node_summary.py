from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
import logging
import json
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

def create_summary_node():
    """Node for summarizing intermediate steps and final answer"""
    logger.info("Initializing summary node")
    llm = create_llm()

    def summary_node(state):
        try:
            logger.debug("Summary Node - State keys: %s", list(state.keys()))

            intermediate_steps = state.get("intermediate_steps", [])
            final_response = state.get("messages", [])[-1].content

            # Prepare descriptions for intermediate steps
            step_descriptions = []
            for action, observation in intermediate_steps:
                step_descriptions.append(f"Tool: {action.tool}")
                step_descriptions.append(f"Input: {json.dumps(action.tool_input, ensure_ascii=False)}")
                step_descriptions.append(f"Observation: {observation}")

            steps_text = "\n".join(step_descriptions)

            # Create prompt for LLM summary
            prompt = f"""
Below are the intermediate analysis steps performed by an AI agent using various tools for stock market analysis:

{steps_text}

And here is the final answer that was returned:
{final_response}

Please:
1. Summarize the entire reasoning process in plain English (layman's terms)
2. Confirm whether the final answer logically follows from the intermediate steps.
3. If any inconsistency is found, suggest a better answer.

Return only the improved or confirmed summary, nothing else.
"""
            
            logger.debug("Summary prompt: %s", prompt)

            summary = llm.invoke(prompt).content
            logger.info("Summary result: %s", summary)

            # Update state
            state["messages"].append(AIMessage(content=summary))
            return state

        except Exception as e:
            logger.error("Error in summary node: %s", str(e), exc_info=True)
            return state

    return summary_node
