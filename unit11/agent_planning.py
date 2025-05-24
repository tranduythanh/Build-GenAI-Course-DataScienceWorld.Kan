from typing import List, Dict, Any, Optional
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings
import os
import logging

logger = logging.getLogger(__name__)

class PlanningAgent:
    """Specialized agent for strategic analysis and planning"""
    
    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.llm: OpenAI = OpenAI(
            api_key=api_key,
            model=self.config.get("llm_model", "gpt-3.5-turbo"),
            temperature=self.config.get("planning_temperature", 0.7)
        )
        
        # Planning state
        self.current_plan: Optional[Dict[str, Any]] = None
        self.execution_history: List[Dict[str, Any]] = []
        
        # System prompt for planning specialist
        planning_prompt = """Bạn là Planning Agent chuyên nghiệp cho phân tích cổ phiếu Việt Nam.

CORE COMPETENCIES:
🧠 **STEP-BACK ANALYSIS**: Phân tích sâu user intent và context
📋 **STRATEGIC PLANNING**: Tạo detailed execution plans với clear steps
🔄 **ADAPTIVE PLANNING**: Điều chỉnh plan dựa trên execution feedback
📊 **CONTEXT AWARENESS**: Hiểu market context và user background

PLANNING METHODOLOGY:
1. **Intent Analysis**: Hiểu chính xác user muốn gì
2. **Context Assessment**: Đánh giá background và constraints
3. **Strategy Formulation**: Tạo step-by-step execution plan
4. **Risk Assessment**: Identify potential issues và mitigation
5. **Success Criteria**: Define clear success metrics

RESPONSE FORMAT:
Always structure responses with:
- 🎯 USER INTENT ANALYSIS
- 📋 EXECUTION PLAN (numbered steps)
- ⚠️ RISK CONSIDERATIONS  
- ✅ SUCCESS CRITERIA
- 🔄 MONITORING POINTS

Focus on practical, actionable plans trong context thị trường Việt Nam."""

        # Initialize memory for planning context
        memory_limit = self.config.get("memory_token_limit", 2000)
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=memory_limit)
        
        logger.info(f"PlanningAgent initialized with memory limit: {memory_limit}")
        
        # Note: Planning agent doesn't need tools initially, 
        # it focuses on analysis and planning
        self.agent: ReActAgent = ReActAgent.from_tools(
            tools=[],  # No tools initially - pure planning focus
            llm=self.llm,
            memory=self.memory,
            verbose=True,
            system_prompt=planning_prompt
        )
    
    def analyze_and_plan(self, user_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform step-back analysis and create strategic plan"""
        
        # Prepare analysis prompt
        analysis_prompt = f"""
USER QUERY: "{user_query}"
CONTEXT: {context if context else "No additional context"}

Thực hiện STEP-BACK ANALYSIS và tạo STRATEGIC PLAN chi tiết.

Phân tích:
1. User thực sự muốn gì? (underlying intent)
2. Loại analysis nào cần thiết?
3. Data requirements và tool sequence
4. Presentation strategy
5. Value-add opportunities

Tạo executable plan với clear steps và success criteria.
"""
        
        try:
            response = self.agent.chat(analysis_prompt)
            
            # Parse and structure the plan
            plan = {
                "query": user_query,
                "analysis": response.response,
                "created_at": str(context.get("timestamp") if context else "now"),
                "status": "created",
                "steps": self._extract_steps(response.response),
                "context": context
            }
            
            # Store current plan
            self.current_plan = plan
            
            return plan
            
        except Exception as e:
            return {
                "error": f"Planning failed: {str(e)}",
                "query": user_query,
                "status": "failed"
            }
    
    def update_plan_progress(self, step_completed: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Update plan based on execution progress"""
        
        if not self.current_plan:
            return {"error": "No active plan to update"}
        
        # Add to execution history
        self.execution_history.append({
            "step": step_completed,
            "results": results,
            "timestamp": "now"
        })
        
        # Get adaptive guidance
        update_prompt = f"""
CURRENT PLAN: {self.current_plan['analysis']}
COMPLETED STEP: {step_completed}
RESULTS: {results}
EXECUTION HISTORY: {self.execution_history}

Dựa trên kết quả này:
1. Plan có cần điều chỉnh không?
2. Bước tiếp theo nên thực hiện như thế nào?
3. Có insights mới nào từ results?
4. Risk assessment có thay đổi không?

Đưa ra adaptive guidance cho execution tiếp theo.
"""
        
        try:
            response = self.agent.chat(update_prompt)
            
            # Update plan with new guidance
            self.current_plan["adaptive_guidance"] = response.response
            self.current_plan["last_updated"] = "now"
            
            return {
                "updated_plan": self.current_plan,
                "guidance": response.response,
                "status": "updated"
            }
            
        except Exception as e:
            return {"error": f"Plan update failed: {str(e)}"}
    
    def get_current_plan(self) -> Optional[Dict[str, Any]]:
        """Get the current active plan"""
        return self.current_plan
    
    def finalize_plan(self, final_results: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize plan and provide comprehensive summary"""
        
        if not self.current_plan:
            return {"error": "No plan to finalize"}
        
        finalization_prompt = f"""
ORIGINAL PLAN: {self.current_plan['analysis']}
EXECUTION HISTORY: {self.execution_history}
FINAL RESULTS: {final_results}

Thực hiện plan finalization:
1. Success assessment against original criteria
2. Key insights và findings summary
3. Value delivered to user
4. Lessons learned cho future planning
5. Final recommendations

Tạo comprehensive summary của toàn bộ planning và execution process.
"""
        
        try:
            response = self.agent.chat(finalization_prompt)
            
            final_summary = {
                "original_plan": self.current_plan,
                "execution_summary": self.execution_history,
                "final_analysis": response.response,
                "status": "completed"
            }
            
            # Reset for next planning cycle
            self.current_plan = None
            self.execution_history = []
            
            return final_summary
            
        except Exception as e:
            return {"error": f"Plan finalization failed: {str(e)}"}
    
    def _extract_steps(self, plan_text: str) -> List[str]:
        """Extract actionable steps from plan text"""
        # Simple extraction - could be more sophisticated
        steps = []
        lines = plan_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered steps or bullet points
            if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '•']):
                steps.append(line)
        
        return steps if steps else ["Execute comprehensive analysis"]
    
    def clear_planning_memory(self) -> None:
        """Clear planning memory and state"""
        try:
            self.agent.reset()
            self.current_plan = None
            self.execution_history = []
        except AttributeError:
            pass 