from multi_agent_system import MultiAgentStockSystem
from agent_config import AgentConfig

config = AgentConfig.from_env()
system = MultiAgentStockSystem(api_key=config.openai_api_key)
response = system.process_query('Tính RSI cho mã VIC với period 14 ngày')
print('=== VIC RSI ANALYSIS ===')
print(response) 