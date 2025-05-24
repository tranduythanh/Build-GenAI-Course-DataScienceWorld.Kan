from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AgentConfig:
    """Configuration class for multi-agent system"""
    
    # API Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Model Configuration
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    
    # Memory Configuration
    memory_token_limit: int = 2000
    max_session_history: int = 100
    
    # Planning Agent Configuration
    planning_max_tokens: int = 800
    planning_temperature: float = 0.7
    
    # Execution Agent Configuration
    execution_max_tokens: int = 600
    execution_temperature: float = 0.5
    
    # Tool Configuration
    stock_price_cache_duration: int = 300  # 5 minutes
    technical_analysis_cache_duration: int = 300  # 5 minutes
    
    # System Configuration
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_health_checks: bool = True
    health_check_interval: int = 3600  # 1 hour
    
    # Performance Configuration
    max_response_time: float = 30.0  # seconds
    max_retries: int = 3
    
    # Vietnamese Stock Configuration
    supported_symbols: list = None
    default_date_range: int = 90  # days
    
    def __post_init__(self):
        if self.supported_symbols is None:
            self.supported_symbols = [
                "HAG", "VIC", "FPT", "VNM", "MSN", 
                "VCB", "BID", "CTG", "TCB", "MBB",
                "VNINDEX"
            ]
    
    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """Create configuration from environment variables"""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
            memory_token_limit=int(os.getenv("MEMORY_TOKEN_LIMIT", "2000")),
            enable_logging=os.getenv("ENABLE_LOGGING", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        if self.llm_temperature < 0 or self.llm_temperature > 2:
            raise ValueError("LLM temperature must be between 0 and 2")
        
        if self.memory_token_limit <= 0:
            raise ValueError("Memory token limit must be positive")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "openai_api_key": "***" if self.openai_api_key else None,  # Hide sensitive data
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "memory_token_limit": self.memory_token_limit,
            "max_session_history": self.max_session_history,
            "planning_max_tokens": self.planning_max_tokens,
            "planning_temperature": self.planning_temperature,
            "execution_max_tokens": self.execution_max_tokens,
            "execution_temperature": self.execution_temperature,
            "stock_price_cache_duration": self.stock_price_cache_duration,
            "technical_analysis_cache_duration": self.technical_analysis_cache_duration,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "enable_health_checks": self.enable_health_checks,
            "health_check_interval": self.health_check_interval,
            "max_response_time": self.max_response_time,
            "max_retries": self.max_retries,
            "supported_symbols": self.supported_symbols,
            "default_date_range": self.default_date_range
        } 