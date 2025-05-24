from multi_agent_system import MultiAgentStockSystem
from agent_config import AgentConfig

def test_multi_agent_system():
    """Test the multi-agent system"""
    print("🔄 Initializing Multi-Agent System...")
    
    config = AgentConfig.from_env()
    system = MultiAgentStockSystem(api_key=config.openai_api_key)
    
    print("✅ System initialized successfully")
    
    # Test 1: VNINDEX analysis
    print("\n" + "="*80)
    print("🚀 TEST 1: Testing with VNINDEX query...")
    print("="*80)
    
    response1 = system.process_query('Phân tích xu hướng giá cổ phiếu VNINDEX')
    
    print("\nSYSTEM RESPONSE 1:")
    print("-"*50)
    print(response1)
    
    # Test 2: VIC technical analysis
    print("\n" + "="*80)
    print("🚀 TEST 2: Testing VIC technical analysis...")
    print("="*80)
    
    response2 = system.process_query('Tính RSI cho mã VIC với period 14 ngày')
    
    print("\nSYSTEM RESPONSE 2:")
    print("-"*50)
    print(response2)
    
    # Display system stats
    stats = system.get_system_stats()
    print("\n" + "="*80)
    print("📊 SYSTEM STATISTICS:")
    print("="*80)
    print(f"Total queries processed: {stats['total_queries']}")
    print(f"Successful queries: {stats['successful_queries']}")
    print(f"Failed queries: {stats['failed_queries']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"Average response time: {stats['average_response_time']:.2f}s")

if __name__ == "__main__":
    test_multi_agent_system() 