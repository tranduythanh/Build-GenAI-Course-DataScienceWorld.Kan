import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config import LLM_MODEL

logger = logging.getLogger(__name__)
logging.getLogger("planner_node").setLevel(logging.DEBUG)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are an expert planning agent for stock technical analysis. Your role is to:
        1. Extract key information from user requests
        2. Plan the analysis steps
        3. Identify required technical indicators
        4. Determine the appropriate time period
        5. Specify what numerical data needs to be collected
        
        Return your analysis in valid JSON format with the following structure:
        - {{ "symbol": string }}
        - {{ "date_range": ["YYYY-MM-DD", "YYYY-MM-DD"] }}
        - {{ "indicators": list of strings }}
        - {{ "steps": list of strings }}
        - {{ "required_data": list of strings describing numerical data needed }}
        
        Ensure all dates are in "YYYY-MM-DD" format.
        Ensure all indicators and data points are quantifiable with specific numerical values.
        """
    ),
    HumanMessagePromptTemplate.from_template("{input}")
])

def create_llm():
    """Create LLM instance"""
    return ChatOpenAI(
        temperature=0,
        model=LLM_MODEL,
        streaming=True,
        verbose=True
    )


def planner_node(llm, state):
    """Process the state through the planner node"""
    try:
        logger.info("Running planner_node")
        
        # Get the current message from the state
        message = state.get("messages", [])[-1]
        logger.info("Planner received message: %s", message.content)

        # Create the prompt with the message content
        try:
            formatted_prompt = prompt.format_messages(input=message.content)
        except Exception as e:
            logger.error("Failed to format prompt: %s", str(e))
            raise ValueError(f"Failed to format prompt: {str(e)}")
        
        logger.debug("Formatted prompt: %s", formatted_prompt)

        # Get the plan response
        plan_response = llm.invoke(formatted_prompt)
        logger.debug("Raw plan response: %s", plan_response.content)

        # Extract JSON from markdown response
        try:
            # First try to find JSON between ```json and ```
            if "```json" in plan_response.content:
                json_str = plan_response.content.split("```json")[1].split("```")[0].strip()
            # If not found, try between ``` and ```
            elif "```" in plan_response.content:
                json_str = plan_response.content.split("```")[1].split("```")[0].strip()
            # If no markdown code blocks, try to find JSON object directly
            else:
                json_str = plan_response.content.strip()
            
            # Remove any leading/trailing whitespace and newlines
            json_str = json_str.strip()
            logger.debug("Extracted JSON string: %s", json_str)

            # Parse and validate the JSON response
            plan_data = json.loads(json_str)
            
            # Validate required keys
            required_keys = ["symbol", "date_range", "indicators", "steps"]
            if not all(key in plan_data for key in required_keys):
                raise ValueError(f"Missing one of required keys in plan: {required_keys}")
            
            # Validate date format
            for date in plan_data["date_range"]:
                if not isinstance(date, str) or len(date.split("-")) != 3:
                    raise ValueError(f"Invalid date format in plan: {date}")
            
            # Create response message
            response_message = AIMessage(
                content=f"Analysis plan created for {plan_data['symbol']}:\n" +
                        f"Period: {plan_data['date_range'][0]} to {plan_data['date_range'][1]}\n" +
                        f"Indicators: {', '.join(plan_data['indicators'])}"
            )
            
            # Update state with plan and response
            new_state = {
                **state,
                "plan": plan_data,
                "action": "calculate_indicators",
                "messages": [response_message],
                "chat_history": state.get("chat_history", []) + [message, response_message]
            }
            
            logger.info("Planner extracted plan: %s", json.dumps(plan_data, indent=2))
            return new_state
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response: %s", str(e))
            raise ValueError("Invalid JSON response from planner")
            
    except Exception as e:
        logger.error("Planner node failed: %s", str(e), exc_info=True)
        raise e

def create_planner_node():
    """Create planner node for the workflow"""
    llm = create_llm()
    return lambda state: planner_node(llm, state)
