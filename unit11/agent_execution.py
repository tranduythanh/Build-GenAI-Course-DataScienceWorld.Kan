from typing import List, Dict, Any, Optional
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings

class ExecutionAgent:
    """Specialized agent for coordinating tool execution based on strategic plans"""
    
    def __init__(self, tools: List[BaseTool], api_key: str, tool_outputs_storage: Dict[str, Any] = None) -> None:
        self.llm: OpenAI = OpenAI(api_key=api_key)
        self.tools = tools
        
        # Execution state
        self.current_execution: Optional[Dict[str, Any]] = None
        self.execution_results: List[Dict[str, Any]] = []
        
        # Reference to shared tool outputs storage
        self.tool_outputs_storage = tool_outputs_storage or {}
        
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
- get_stock_price: Lấy dữ liệu giá cổ phiếu Việt Nam với AI analysis
- calculate_technical_indicators: Phân tích kỹ thuật (SMA, RSI, MACD, Bollinger Bands)

MANDATORY EXECUTION RULES:
🔥 **ALWAYS USE TOOLS**: Bạn PHẢI sử dụng available tools để lấy dữ liệu thực
🔥 **NO THEORETICAL RESPONSES**: KHÔNG BAO GIỜ trả lời lý thuyết mà không có dữ liệu thực
🔥 **TOOL-DRIVEN ANALYSIS**: Mọi phân tích phải dựa trên kết quả từ tools
🔥 **DATA REQUIREMENT**: Với mọi câu hỏi về stock, BẮT BUỘC phải gọi get_stock_price
🔥 **TECHNICAL ANALYSIS**: Với mọi yêu cầu technical indicators, BẮT BUỘC phải gọi calculate_technical_indicators

EXECUTION PRINCIPLES:
- ALWAYS call appropriate tools FIRST để lấy dữ liệu
- Follow plan sequence chính xác với REAL DATA
- Validate results at each step
- Provide detailed progress feedback based on ACTUAL RESULTS
- Synthesize comprehensive final output from TOOL OUTPUTS
- Handle errors gracefully với fallback strategies

RESPONSE FORMAT:
1. Determine required tools based on query/plan
2. Execute tools with appropriate parameters 
3. INCLUDE FULL TOOL OUTPUTS trong response (bảng stock prices + technical indicators)
4. Analyze ACTUAL results from tools
5. Provide insights based on REAL DATA với references đến cụ thể từng chỉ số
6. Give recommendations based on CONCRETE FINDINGS

🔥 CRITICAL OUTPUT REQUIREMENTS:
- BẮT BUỘC copy nguyên văn bảng "Recent 10 Trading Days" từ get_stock_price
- BẮT BUỘC copy nguyên văn bảng "Recent 5 Days Technical Indicators" từ calculate_technical_indicators  
- BẮT BUỘC show Latest Technical Indicators với số liệu cụ thể
- User phải thấy được RAW DATA, không chỉ summary

Focus on efficient, accurate execution theo strategic plan với FULL DATA DISPLAY từ tools."""

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
        
        # Hook into agent's tool execution to capture outputs
        self._setup_tool_capture()
    
    def _setup_tool_capture(self) -> None:
        """Setup tool output capture by wrapping tools"""
        for tool in self.tools:
            original_call = tool.__call__
            
            def captured_call(input_data, tool_ref=tool, original_method=original_call):
                # Call original tool
                result = original_method(input_data)
                
                # Store the output for later use
                tool_name = tool_ref.metadata.name
                self.tool_outputs_storage[tool_name] = {
                    "input": input_data,
                    "output": result,
                    "content": result.content if hasattr(result, 'content') else str(result)
                }
                
                return result
            
            # Replace the tool's call method
            tool.__call__ = captured_call
    
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

🔥 MANDATORY EXECUTION REQUIREMENTS:
1. BẮT BUỘC phải sử dụng get_stock_price tool để lấy dữ liệu cổ phiếu thực
2. BẮT BUỘC phải sử dụng calculate_technical_indicators cho MỌI câu hỏi technical analysis
3. KHÔNG BAO GIỜ trả lời lý thuyết mà không có dữ liệu từ tools
4. Mọi phân tích phải dựa trên kết quả THỰC từ tool execution

⚠️ TECHNICAL ANALYSIS DETECTION:
Keywords requiring calculate_technical_indicators: RSI, SMA, MACD, Bollinger, chỉ số kỹ thuật, technical analysis, phân tích kỹ thuật

EXECUTION STEPS:
1. Identify symbol từ user query (VIC, FPT, VNM, HAG, MSN, VNINDEX, etc.)
2. MANDATORY STEP 1: call get_stock_price với symbol và date range → Lấy dữ liệu 10 ngày gần nhất
3. MANDATORY STEP 2: call calculate_technical_indicators với cùng symbol → Lấy RSI, SMA, MACD, Bollinger Bands
4. ANALYZE & SYNTHESIZE: Kết hợp kết quả từ cả hai tools
5. COMPREHENSIVE INSIGHTS: Phân tích price trends + technical signals + recommendations  
6. SPECIFIC REFERENCES: Đề cập cụ thể đến từng chỉ số TA trong analysis

🔥 LUÔN LUÔN GỌI CẢ HAI TOOLS:
- get_stock_price → Stock price data + 10 recent days
- calculate_technical_indicators → RSI, SMA, MACD, Bollinger Bands

📊 MANDATORY OUTPUT INCLUSION:
- Copy bảng "Recent 10 Trading Days" từ get_stock_price output
- Copy bảng "Recent 5 Days Technical Indicators" từ calculate_technical_indicators output
- Copy "Latest Technical Indicators" với tất cả số liệu (Price, SMA, RSI, MACD)
- Show performance metrics nếu có

📝 REQUIRED RESPONSE FORMAT:
```
## 📊 STOCK PRICE DATA
[Copy nguyên văn bảng Recent 10 Trading Days từ get_stock_price]

## 📈 TECHNICAL ANALYSIS  
[Copy nguyên văn Latest Technical Indicators + Recent 5 Days Technical Indicators từ calculate_technical_indicators]

## 💡 ANALYSIS & INSIGHTS
[Phân tích dựa trên dữ liệu cụ thể trên]
```

🚀 BẮT ĐẦU EXECUTION - CALL BOTH TOOLS VÀ INCLUDE FULL DATA THEO FORMAT!
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