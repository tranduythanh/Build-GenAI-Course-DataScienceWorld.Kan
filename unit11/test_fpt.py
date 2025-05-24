from multi_agent_system import MultiAgentStockSystem
from agent_config import AgentConfig

def test_fpt_analysis():
    """Test FPT analysis with specific date range"""
    print("🔄 Initializing Multi-Agent System...")
    
    config = AgentConfig.from_env()
    system = MultiAgentStockSystem(api_key=config.openai_api_key)
    
    print("✅ System initialized successfully")
    print("🚀 Testing FPT analysis with date range...")
    
    response = system.process_query('Get stock price for FPT from 2024-01-01 to 2024-06-30')
    
    print("="*120)
    print("🎯 FPT STOCK ANALYSIS RESULT:")
    print("="*120)
    print(response)
    print("="*120)

if __name__ == "__main__":
    test_fpt_analysis() 