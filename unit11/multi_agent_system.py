from typing import List, Dict, Any, Optional
from llama_index.core.tools import BaseTool
from agent_planning import PlanningAgent
from agent_execution import ExecutionAgent
from llama_tool_stock_price import StockPriceTool
from llama_tool_technical_analysis import TechnicalAnalysisTool
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiAgentStockSystem:
    """Multi-agent system for comprehensive Vietnamese stock analysis"""
    
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        
        # Initialize tools
        self.tools: List[BaseTool] = [
            StockPriceTool(),
            TechnicalAnalysisTool()
        ]
        
        try:
            # Initialize specialized agents
            self.planning_agent = PlanningAgent(api_key=api_key)
            self.execution_agent = ExecutionAgent(tools=self.tools, api_key=api_key)
            logger.info("Multi-agent system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            raise
        
        # System state
        self.current_session: Optional[Dict[str, Any]] = None
        self.session_history: List[Dict[str, Any]] = []
        self.system_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_response_time": 0.0
        }
    
    def process_query(self, user_query: str, context: Dict[str, Any] = None) -> str:
        """Process user query using multi-agent approach"""
        
        start_time = datetime.now()
        self.system_metrics["total_queries"] += 1
        
        logger.info(f"Processing query: {user_query[:50]}...")
        
        try:
            # Step 1: Planning Phase
            logger.info("🧠 PLANNING PHASE: Analyzing query and creating strategic plan...")
            plan = self.planning_agent.analyze_and_plan(user_query, context)
            
            if plan.get("status") == "failed":
                error_msg = f"❌ Planning failed: {plan.get('error', 'Unknown error')}"
                logger.error(error_msg)
                self.system_metrics["failed_queries"] += 1
                return error_msg
            
            logger.info("✅ Planning completed successfully")
            
            # Step 2: Execution Phase  
            logger.info("🚀 EXECUTION PHASE: Executing plan with specialized tools...")
            execution_result = self.execution_agent.execute_plan(plan, user_query)
            
            if execution_result.get("status") == "failed":
                error_msg = f"❌ Execution failed: {execution_result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                self.system_metrics["failed_queries"] += 1
                return error_msg
                
            logger.info("✅ Execution completed successfully")
            
            # Step 3: Planning Feedback & Finalization
            logger.info("🔄 FINALIZATION PHASE: Gathering feedback and finalizing analysis...")
            final_summary = self.planning_agent.finalize_plan(execution_result)
            
            # Calculate response time
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Create session record
            session = {
                "query": user_query,
                "plan": plan,
                "execution": execution_result,
                "final_summary": final_summary,
                "timestamp": start_time.isoformat(),
                "response_time": response_time,
                "status": "completed"
            }
            
            self.current_session = session
            self.session_history.append(session)
            
            # Update metrics
            self.system_metrics["successful_queries"] += 1
            self._update_average_response_time(response_time)
            
            # Format comprehensive response
            response = self._format_comprehensive_response(session)
            
            logger.info(f"✅ Query processed successfully in {response_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Multi-agent system error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            error_msg = f"❌ Multi-agent system error: {str(e)}"
            self.system_metrics["failed_queries"] += 1
            
            # Create error session
            error_session = {
                "query": user_query,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": start_time.isoformat(),
                "status": "failed"
            }
            self.session_history.append(error_session)
            
            return error_msg
    
    def _format_comprehensive_response(self, session: Dict[str, Any]) -> str:
        """Format comprehensive response from multi-agent analysis"""
        
        plan_analysis = session.get("plan", {}).get("analysis", "No plan analysis")
        execution_response = session.get("execution", {}).get("execution_response", "No execution results")
        final_analysis = session.get("final_summary", {}).get("final_analysis", "No final summary")
        response_time = session.get("response_time", 0)
        
        response = f"""
🎯 **COMPREHENSIVE VIETNAMESE STOCK ANALYSIS**

**📝 Query**: {session.get("query", "")}

---

## 🧠 **STRATEGIC ANALYSIS & PLANNING**

{plan_analysis}

---

## 🚀 **EXECUTION RESULTS**

{execution_response}

---

## 📋 **EXECUTIVE SUMMARY & RECOMMENDATIONS**

{final_analysis}

---
✅ **Analysis completed using specialized multi-agent approach**
📊 **Planning Agent**: Strategic analysis and step-by-step planning
🔧 **Execution Agent**: Tool coordination and data synthesis
💡 **Result**: Comprehensive insights for Vietnamese stock market
⚡ **Processing Time**: {response_time:.2f} seconds

*Powered by Multi-Agent Stock Analysis System*
        """
        
        return response.strip()
    
    def _update_average_response_time(self, new_time: float) -> None:
        """Update running average of response times"""
        total_successful = self.system_metrics["successful_queries"]
        current_avg = self.system_metrics["average_response_time"]
        
        # Running average calculation
        new_avg = ((current_avg * (total_successful - 1)) + new_time) / total_successful
        self.system_metrics["average_response_time"] = new_avg
    
    def get_planning_insights(self, query: str) -> Dict[str, Any]:
        """Get planning insights without full execution"""
        logger.info(f"Getting planning insights for: {query[:50]}...")
        try:
            return self.planning_agent.analyze_and_plan(query)
        except Exception as e:
            logger.error(f"Planning insights failed: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def execute_custom_plan(self, plan: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Execute a custom plan"""
        logger.info(f"Executing custom plan for: {query[:50]}...")
        try:
            return self.execution_agent.execute_plan(plan, query)
        except Exception as e:
            logger.error(f"Custom plan execution failed: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def get_session_status(self) -> Optional[Dict[str, Any]]:
        """Get current session status"""
        return self.current_session
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get history of all sessions"""
        return self.session_history
    
    def clear_system_memory(self) -> None:
        """Clear all agent memories and system state"""
        logger.info("Clearing system memory...")
        try:
            self.planning_agent.clear_planning_memory()
            self.execution_agent.clear_execution_memory()
            self.current_session = None
            self.session_history = []
            logger.info("✅ System memory cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear system memory: {str(e)}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        total_sessions = len(self.session_history)
        successful_sessions = len([s for s in self.session_history if s.get("status") == "completed"])
        
        return {
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "failed_sessions": self.system_metrics["failed_queries"],
            "success_rate": (successful_sessions / total_sessions * 100) if total_sessions > 0 else 0,
            "average_response_time": self.system_metrics["average_response_time"],
            "planning_agent_active": self.planning_agent.get_current_plan() is not None,
            "execution_agent_active": self.execution_agent.get_execution_status() is not None,
            "available_tools": len(self.tools),
            "system_health": "healthy" if self.system_metrics["failed_queries"] < self.system_metrics["total_queries"] * 0.1 else "degraded"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        logger.info("Performing system health check...")
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "agents": {},
            "tools": {},
            "overall_status": "healthy"
        }
        
        # Check agents
        try:
            # Test planning agent
            test_plan = self.planning_agent.analyze_and_plan("test query")
            health_status["agents"]["planning"] = "healthy" if test_plan.get("status") != "failed" else "unhealthy"
        except Exception as e:
            health_status["agents"]["planning"] = f"error: {str(e)}"
            health_status["overall_status"] = "degraded"
        
        try:
            # Test execution agent
            exec_status = self.execution_agent.get_execution_status()
            health_status["agents"]["execution"] = "healthy"
        except Exception as e:
            health_status["agents"]["execution"] = f"error: {str(e)}"
            health_status["overall_status"] = "degraded"
        
        # Check tools
        for i, tool in enumerate(self.tools):
            try:
                tool_name = tool.metadata.name
                health_status["tools"][tool_name] = "healthy"
            except Exception as e:
                health_status["tools"][f"tool_{i}"] = f"error: {str(e)}"
                health_status["overall_status"] = "degraded"
        
        logger.info(f"Health check completed: {health_status['overall_status']}")
        return health_status 