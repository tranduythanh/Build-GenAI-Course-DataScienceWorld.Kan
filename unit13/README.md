# Bitcoin Q&A System - Hệ thống Hỏi Đáp Bitcoin (MVP với LangGraph)

## 📋 Tổng quan dự án

MVP hệ thống hỏi đáp thông minh về Bitcoin sử dụng LangGraph Agent với GPT-4.1-mini, tích hợp các tools chuyên biệt để phân tích technical indicators và cung cấp thông tin thị trường crypto theo yêu cầu.

## 🎯 Mục tiêu MVP

1. **Agent-based Q&A**: Hệ thống agent thông minh trả lời câu hỏi về Bitcoin
2. **Tool Integration**: Agent sử dụng tools chuyên biệt để thu thập và phân tích dữ liệu
3. **Real-time Data**: Cập nhật dữ liệu thị trường theo yêu cầu
4. **Technical Analysis**: Tính toán và giải thích các chỉ số kỹ thuật
5. **Smart Query Processing**: Sử dụng step-back strategy và question rephrasing
6. **Session Context Management**: Lưu trữ và tái sử dụng kết quả trung gian

## 🏗️ Kiến trúc MVP với LangGraph

### Core Components
```
├── agents/
│   ├── bitcoin_qa_agent.py       # Main LangGraph agent
│   ├── tools/                    # Agent tools
│   │   ├── price_fetcher.py      # Tool lấy giá Bitcoin
│   │   ├── technical_analyzer.py # Tool phân tích kỹ thuật
│   │   ├── news_fetcher.py       # Tool lấy tin tức
│   │   └── market_data.py        # Tool dữ liệu thị trường
│   └── workflows/
│       └── qa_workflow.py        # LangGraph workflow định nghĩa
├── data/
│   ├── bitcoin_knowledge/       # Static knowledge base
│   ├── cache/                   # Runtime cache
│   └── sessions/                # Session context storage
└── app.py
```

### LangGraph Agent Tools

- **PriceFetcher Tool**: Lấy giá Bitcoin hiện tại và lịch sử (package: ccxt)
- **TechnicalAnalyzer Tool**: Tính toán các chỉ số kỹ thuật (RSI, MACD, SMA, v.v.)
- **NewsFetcher Tool**: Lấy tin tức Bitcoin mới nhất (package: feedparser)
- **MarketData Tool**: Dữ liệu tổng quan thị trường (dominance, sentiment, v.v.)
- **Tavily Search Tool**: Dùng cho các logic cần search internet (tin tức, thông tin mới, v.v.)

## 📊 Technical Indicators được hỗ trợ trong MVP

### 1. Essential Indicators
- **Simple Moving Average (SMA)**: 20, 50, 200 periods
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Price volatility bands

### 2. Support/Resistance
- **Current Support/Resistance levels**
- **Key price levels based on historical data**

## 🤖 LangGraph Workflow

### Agent Decision Tree
```
User Question → Agent Router → Tool Selection → Data Processing → Response Generation
                     ↓
                Question Classification:
                ├── Basic Info → KnowledgeBase Tool
                ├── Price Query → PriceFetcher Tool  
                ├── Technical Analysis → TechnicalAnalyzer Tool
                ├── Market Sentiment → NewsFetcher + MarketData Tools
                └── Complex Analysis → Multi-tool combination
```

### Example Workflows

#### 1. Price Analysis Workflow
```
User: "Giá Bitcoin hiện tại và RSI như thế nào?"
├── PriceFetcher Tool → Current BTC price
├── TechnicalAnalyzer Tool → Calculate RSI
└── Response: Combine price + RSI analysis + interpretation
```

#### 2. Market Overview Workflow  
```
User: "Tình hình thị trường Bitcoin hôm nay?"
├── PriceFetcher Tool → BTC price movement
├── MarketData Tool → Market dominance, volume
├── NewsFetcher Tool → Recent news sentiment
└── Response: Comprehensive market summary
```

## 💻 Tech Stack cho MVP

### Core Framework
- **LangGraph**: Agent workflow orchestration
- **GPT-4.1-mini**: Primary LLM for reasoning and responses
- **Python 3.9.12**: Backend language
- **Streamlit**: Simple web interface for testing

### Data & Analysis
- **yfinance/ccxt**: Cryptocurrency data fetching
- **pandas**: Data manipulation
- **talib**: Technical analysis calculations
- **requests**: API calls for external data
- **tavily**: Internet search (bắt buộc dùng tavily cho mọi logic cần search internet)

### Infrastructure
- **Redis**: Simple caching (optional for MVP)
- **SQLite**: Lightweight database for cache

## Development Setup

### Install Poetry and Dependencies
```bash
pip install --user poetry
cd unit13
poetry install --no-root --with dev
```

### Run Tests and Type Checks
```bash
poetry run flake8 agents app.py tests __init__.py
poetry run mypy --ignore-missing-imports --explicit-package-bases agents app.py tests __init__.py
make test
```
