from typing import List, Dict, Any, Optional
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings

class ExecutionAgent:
    """Specialized agent for coordinating tool execution based on strategic plans"""
    
    def __init__(self, tools: List[BaseTool], api_key: str) -> None:
        self.llm: OpenAI = OpenAI(api_key=api_key)
        self.tools = tools
        
        # Execution state
        self.current_execution: Optional[Dict[str, Any]] = None
        self.execution_results: List[Dict[str, Any]] = []
        
        # System prompt for execution specialist
        execution_prompt = """Bạn là Execution Agent chuyên nghiệp cho việc thực hiện phân tích cổ phiếu Việt Nam.

CORE COMPETENCIES:
🚀 **PLAN EXECUTION**: Thực hiện strategic plans một cách systematic
🔧 **TOOL COORDINATION**: Coordinate multiple tools hiệu quả
📊 **DATA SYNTHESIS**: Tổng hợp results từ multiple sources
🎯 **GOAL ACHIEVEMENT**: Focus vào achieving plan objectives

EXECUTION METHODOLOGY:
1. **Plan Interpretation**: Hiểu rõ plan requirements và sequence
2. **Tool Selection**: Chọn đúng tools cho từng step
3. **Parameter Optimization**: Optimize tool parameters cho best results
4. **Result Integration**: Tích hợp và synthesize multiple results
5. **Quality Assurance**: Ensure results meet plan criteria

AVAILABLE TOOLS:
- get_stock_price: Lấy dữ liệu giá cổ phiếu Việt Nam
- calculate_technical_indicators: Phân tích kỹ thuật (SMA, RSI, MACD, Bollinger Bands)

EXECUTION PRINCIPLES:
- Follow plan sequence chính xác
- Validate results at each step
- Provide detailed progress feedback
- Synthesize comprehensive final output
- Handle errors gracefully với fallback strategies

Focus on efficient, accurate execution theo strategic plan."""

        # Initialize memory for execution context
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
        
        # Initialize agent with tools
        self.agent: ReActAgent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            memory=self.memory,
            verbose=True,
            system_prompt=execution_prompt
        )
    
    def execute_plan(self, plan: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Execute a strategic plan step by step"""
        
        if not plan or plan.get("status") == "failed":
            return {"error": "Invalid or failed plan provided"}
        
        # Initialize execution
        self.current_execution = {
            "plan": plan,
            "query": user_query,
            "status": "executing",
            "current_step": 0,
            "results": []
        }
        
        # Execute based on plan
        execution_prompt = f"""
STRATEGIC PLAN TO EXECUTE:
{plan.get('analysis', '')}

USER QUERY: "{user_query}"
EXTRACTED STEPS: {plan.get('steps', [])}

Thực hiện plan này step by step:

1. Phân tích plan để identify required tools và sequence
2. Execute tools với appropriate parameters
3. Tại mỗi step, validate results và provide progress feedback
4. Synthesize final comprehensive response

Bắt đầu execution theo plan. Use available tools systematically.
"""
        
        try:
            response = self.agent.chat(execution_prompt)
            
            # Store execution results
            execution_result = {
                "plan_executed": plan,
                "execution_response": response.response,
                "status": "completed",
                "query": user_query
            }
            
            self.execution_results.append(execution_result)
            self.current_execution["status"] = "completed"
            
            return execution_result
            
        except Exception as e:
            error_result = {
                "error": f"Execution failed: {str(e)}",
                "plan": plan,
                "query": user_query,
                "status": "failed"
            }
            
            self.execution_results.append(error_result)
            if self.current_execution:
                self.current_execution["status"] = "failed"
            
            return error_result
    
    def execute_step(self, step_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a single step with context"""
        
        step_prompt = f"""
STEP TO EXECUTE: {step_description}
CONTEXT: {context if context else "No additional context"}
PREVIOUS RESULTS: {self.execution_results[-3:] if self.execution_results else "None"}

Execute this specific step:
1. Analyze step requirements
2. Select appropriate tool(s)
3. Execute with optimal parameters
4. Validate and format results
5. Provide step completion summary

Focus on accurate execution của step này.
"""
        
        try:
            response = self.agent.chat(step_prompt)
            
            step_result = {
                "step": step_description,
                "result": response.response,
                "context": context,
                "status": "completed"
            }
            
            # Update current execution if active
            if self.current_execution:
                self.current_execution["results"].append(step_result)
                self.current_execution["current_step"] += 1
            
            return step_result
            
        except Exception as e:
            return {
                "step": step_description,
                "error": f"Step execution failed: {str(e)}",
                "status": "failed"
            }
    
    def synthesize_results(self, results: List[Dict[str, Any]], original_query: str) -> Dict[str, Any]:
        """Synthesize multiple execution results into comprehensive response"""
        
        synthesis_prompt = f"""
ORIGINAL USER QUERY: "{original_query}"
EXECUTION RESULTS: {results}

Synthesize comprehensive final response:

1. **EXECUTIVE SUMMARY**: Key findings và insights
2. **DETAILED ANALYSIS**: Breakdown of all results
3. **PRACTICAL RECOMMENDATIONS**: Actionable advice cho user
4. **RISK ASSESSMENT**: Important caveats và limitations
5. **NEXT STEPS**: Suggested follow-up actions

Tạo professional, comprehensive response tích hợp all execution results.
Focus on value delivery cho Vietnamese stock market context.
"""
        
        try:
            response = self.agent.chat(synthesis_prompt)
            
            return {
                "synthesis": response.response,
                "original_query": original_query,
                "results_count": len(results),
                "status": "completed"
            }
            
        except Exception as e:
            return {
                "error": f"Synthesis failed: {str(e)}",
                "original_query": original_query,
                "status": "failed"
            }
    
    def get_execution_status(self) -> Optional[Dict[str, Any]]:
        """Get current execution status"""
        return self.current_execution
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get history of all executions"""
        return self.execution_results
    
    def clear_execution_memory(self) -> None:
        """Clear execution memory and state"""
        try:
            self.agent.reset()
            self.current_execution = None
            self.execution_results = []
        except AttributeError:
            pass 