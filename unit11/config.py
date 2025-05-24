from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LlamaIndex Configuration
LLAMA_CONFIG: Dict[str, Any] = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 1000,
    "memory_token_limit": 1500,
    "verbose": True
}

# Technical Analysis Configuration
TECH_ANALYSIS_CONFIG: Dict[str, Any] = {
    "sma_period": 20,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2
}

# Data Collection Configuration
DATA_COLLECTION_CONFIG: Dict[str, Any] = {
    "cache_dir": "data/cache",
    "max_retries": 3,
    "timeout": 30
}

# Report Generation Configuration
REPORT_CONFIG: Dict[str, Any] = {
    "template_dir": "templates",
    "output_dir": "reports",
    "supported_formats": ["json", "html", "pdf"]
} 