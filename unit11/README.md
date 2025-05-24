# Stock Quant

A comprehensive financial analysis and trading toolkit built with Python. This application provides various tools for stock market analysis, including price data collection, technical indicators, and AI-powered analysis using LlamaIndex.

## Features

### 1. Stock Data Collection
- Real-time and historical stock price data using VNQuant library
- Custom date ranges for data retrieval
- Support for multiple stock symbols simultaneously
- Automatic data processing with pandas
- Intelligent caching system for improved performance
- Date validation with minimum 30-day periods

### 2. Technical Analysis
- Calculate multiple technical indicators using pandas-ta
- Supported indicators:
  - Simple Moving Average (SMA) with custom parameters
  - Relative Strength Index (RSI) with custom periods
  - Moving Average Convergence Divergence (MACD) with fast, slow, signal parameters
  - Bollinger Bands with standard deviation and period customization
- Automatic data handling and error processing
- Support for multiple timeframes

### 3. AI-Powered Analysis
- Natural language processing for queries using LlamaIndex
- Advanced agent system with specialized tools:
  - Stock Price Tool: Retrieve and process data from VNQuant
  - Technical Analysis Tool: Calculate technical indicators
  - Data Collector Tool: Web scraping and data collection
  - Finance Report Tool: Generate comprehensive financial reports
- Interactive chat history for contextual responses
- Memory management for conversation context
- Query engine for complex financial analysis

### 4. Financial Reporting
- Comprehensive financial reports using VNQuant data
- Support for multiple report types:
  - Financial statements
  - Business performance reports
  - Cash flow analysis
  - Basic financial indicators
- Export capabilities in multiple formats

## Architecture

The application follows a modular architecture with the following components:

### 1. Core Data Layer
- **`StockDataCollector`**: Base class for stock data fetching using VNQuant
- **`DataCollector`**: Enhanced version with caching and additional features
- Inheritance-based design for code reuse and maintainability

### 2. LlamaIndex Integration
- **`StockQuantAgent`**: Main agent powered by OpenAI and LlamaIndex
- **Service Context**: Manages LLM, memory, and tool integration
- **Query Engine**: Handles complex multi-step queries
- **Response Synthesizer**: Generates coherent responses

### 3. Tool System
- **Stock Price Tool**: Fetches stock data with caching support
- **Technical Analysis Tool**: Calculates indicators using pandas-ta
- **Data Collector Tool**: Web-based data collection
- **Finance Report Tool**: Generates financial analysis reports

### 4. Configuration & Utilities
- **Config System**: Centralized configuration management
- **Logger**: Comprehensive logging for all components
- **Type Definitions**: Strong typing for better code quality

## Installation

1. Clone repository:
```bash
git clone <repository-url>
cd stock-quant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file and configure environment variables:
```
OPENAI_API_KEY=your_openai_api_key
VNQUANT_API_KEY=your_vnquant_api_key
```

4. Run the application:
```bash
streamlit run app.py
```

## File Structure

```
stock-quant/
├── app.py                    # Main Streamlit application entry point
├── llama_agent.py           # LlamaIndex-powered agent implementation
├── llama_tools.py           # Tool implementations for LlamaIndex
├── llama_data_collector.py  # Enhanced data collector with caching
├── llama_types.py           # Type definitions for the system
├── data_collector.py        # Base stock data collector
├── tech_analysis.py         # Technical analysis functions
├── finance_report.py        # Financial report generation
├── config.py                # Configuration settings
├── logger.py                # Logging setup and utilities
├── example_stock_data.py    # Example usage of data collector
├── requirements.txt         # Python dependencies
└── README.md                # Documentation
```

## Usage Examples

### Basic Data Collection
```python
from data_collector import StockDataCollector

collector = StockDataCollector()
data = collector.fetch_stock_data('VIC', '2024-01-01', '2024-12-31')
print(data.head())
```

### Enhanced Data Collection with Caching
```python
from llama_data_collector import DataCollector

collector = DataCollector(api_key="your_api_key")
data = collector.get_stock_data('VNM', '2024-01-01', '2024-12-31', use_cache=True)
```

### Technical Analysis
```python
from tech_analysis import TechAna

analyzer = TechAna()
rsi_result = analyzer.analyze(data, 'RSI', {'length': 14})
```

### AI-Powered Queries
Run the Streamlit app and try these queries:
- "Show me the stock price of VNM in the last 10 days"
- "Calculate the RSI for VIC over the last month"
- "What are the Bollinger Bands for HAG?"
- "Compare VNM and FPT performance"
- "Generate a technical analysis report for VIC"

## Key Features

### 1. Intelligent Caching
- Automatic caching of stock data to improve performance
- Configurable cache duration and cleanup
- Cache invalidation strategies

### 2. Error Handling
- Comprehensive error handling throughout the system
- Graceful degradation for missing data
- Detailed logging for debugging

### 3. Type Safety
- Strong typing with Python type hints
- Custom type definitions for better code quality
- Runtime type checking where appropriate

### 4. Modular Design
- Clean separation of concerns
- Inheritance-based code reuse
- Easy extensibility for new features

### 5. AI Integration
- LlamaIndex-powered natural language interface
- Context-aware conversations
- Multi-tool coordination for complex queries

## Configuration

The system uses a centralized configuration approach:

```python
# Configuration categories
LLAMA_CONFIG          # LlamaIndex settings
TECH_ANALYSIS_CONFIG  # Technical analysis parameters
DATA_COLLECTION_CONFIG # Caching and API settings
REPORT_CONFIG         # Report generation settings
```

## Dependencies

Key dependencies include:
- `streamlit`: Web interface
- `llama-index`: AI agent framework
- `vnquant`: Vietnamese stock data
- `pandas`: Data manipulation
- `pandas-ta`: Technical analysis
- `python-dotenv`: Environment management

## Notes

- The application uses VNQuant library for Vietnamese stock market data
- Technical indicators are calculated using the pandas-ta library
- User interface is built with Streamlit
- AI system uses LlamaIndex with OpenAI for natural language processing
- Caching system improves performance for repeated queries
- The system supports both English and Vietnamese for user interactions
