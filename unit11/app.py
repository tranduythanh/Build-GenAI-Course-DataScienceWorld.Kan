from typing import List, Optional
import streamlit as st
import os
from dotenv import load_dotenv
from multi_agent_system import MultiAgentStockSystem
from agent_config import AgentConfig

# Load environment variables
load_dotenv()

# Initialize configuration
@st.cache_resource
def initialize_system():
    """Initialize the multi-agent system with caching"""
    config = AgentConfig.from_env()
    config.validate()
    return MultiAgentStockSystem(api_key=config.openai_api_key), config

multi_agent_system, system_config = initialize_system()

# Streamlit UI
st.title("📈 Vietnamese Stock Analysis Assistant")
st.markdown("### Ask me about Vietnamese stocks!")

# Add sample queries
st.markdown("""
**Sample questions you can ask:**
- "Xin stock data của mã HAG"
- "Xin technical analysis của mã VIC"
- "Get stock price for FPT from 2024-01-01 to 2024-06-30"
- "Calculate technical indicators for MSN"
""")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Hỏi về cổ phiếu Việt Nam (VD: Xin stock data của mã HAG)"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get multi-agent response
    with st.chat_message("assistant"):
        try:
            response: str = multi_agent_system.process_query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg: str = f"Error: {str(e)}"
            st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        multi_agent_system.clear_system_memory()
        st.rerun()
    
    # Multi-agent system stats
    stats = multi_agent_system.get_system_stats()
    st.header("🤖 Multi-Agent Stats")
    st.write(f"📊 Total Sessions: {stats['total_sessions']}")
    st.write(f"✅ Success Rate: {stats['success_rate']:.1f}%")
    st.write(f"⚡ Avg Response Time: {stats['average_response_time']:.2f}s")
    st.write(f"🏥 System Health: {stats['system_health'].title()}")
    st.write(f"🧠 Planning Agent: {'Active' if stats['planning_agent_active'] else 'Idle'}")
    st.write(f"🚀 Execution Agent: {'Active' if stats['execution_agent_active'] else 'Idle'}")
    
    # Health check button
    if st.button("🔍 Run Health Check"):
        with st.spinner("Running health check..."):
            health = multi_agent_system.health_check()
            st.json(health)
    
    # Configuration display
    if st.checkbox("🔧 Show Configuration"):
        st.subheader("System Configuration")
        config_dict = system_config.to_dict()
        st.json(config_dict)
    
    # Display chat history
    st.header("Chat History")
    for message in st.session_state.messages:
        st.text(f"{message['role']}: {message['content']}")

# Information about multi-agent system
with st.sidebar:
    st.header("🤖 Multi-Agent Architecture")
    st.write("🧠 **Planning Agent**: Specialized trong step-back analysis và strategic planning")
    st.write("🚀 **Execution Agent**: Chuyên coordinate tools và synthesize results")
    st.write("📊 **Stock Tools**: Get price data và technical analysis")
    
    st.header("⚡ Multi-Agent Advantages")
    st.write("✅ **Specialized Reasoning**: Mỗi agent có expertise riêng")
    st.write("✅ **Memory Management**: Separate memories cho planning vs execution")
    st.write("✅ **Adaptive Planning**: Plan có thể adjust based on execution feedback")
    st.write("✅ **Better Coordination**: Clear separation of concerns")
    st.write("✅ **State Tracking**: Detailed tracking của plan progress")
    
    st.header("💡 Vietnamese Stock Symbols")
    st.write("- HAG (HAGL Agrico)")
    st.write("- VIC (Vingroup)")
    st.write("- FPT (FPT Corporation)")
    st.write("- VNM (Vinamilk)")
    st.write("- MSN (Masan Group)")
    st.write("- VNINDEX (VN-Index)") 