# Bitcoin Trading System - Hệ thống Giao dịch Bitcoin (CrewAI)

## 📋 Tổng quan dự án

Dự án cung cấp một hệ thống giao dịch Bitcoin đơn giản sử dụng CrewAI với 3 agent chuyên biệt. Mỗi agent có một nhiệm vụ cụ thể và làm việc cùng nhau để đưa ra quyết định giao dịch.

## 🎯 Cài đặt

Yêu cầu Python 3.11.12.

```bash
uv pip install crewai==0.130.0 crewai_tools==0.47.1 langchain_community==0.3.25
```

## 🤖 Các Agent trong Hệ thống

### Information Agent
- Thu thập giá Bitcoin và vàng
- Cập nhật tin tức thị trường
- Sử dụng SerperDevTool để tìm kiếm thông tin

### Technical Analysis Agent
- Phân tích các chỉ số kỹ thuật cơ bản:
  - SMA (Simple Moving Average)
  - RSI (Relative Strength Index)
  - BB (Bollinger Bands)
  - MACD (Moving Average Convergence Divergence)
- Đưa ra dự báo xu hướng

### Trading Agent
- Quản lý danh mục đầu tư
- Đưa ra quyết định mua/bán dựa trên:
  - Phân tích kỹ thuật
  - Thông tin thị trường
- Xác định khối lượng giao dịch

## 📁 Cấu trúc dự án
```
├── agents/
│   ├── information_agent.py
│   ├── analysis_agent.py
│   ├── trading_agent.py
│   └── tools/
│       ├── price_fetcher.py
│       └── technical_analyzer.py
├── app.py
└── requirements.txt
```

## 🔄 Luồng hoạt động

```mermaid
graph TD
    A["User Input"] --> B["CrewAI Orchestrator"]
    B --> C["Information Agent"]
    B --> D["Technical Analysis Agent"]
    B --> E["Trading Agent"]
    C --> F["Market Data"]
    D --> G["Technical Analysis"]
    E --> H["Trading Decision"]
```

## 🚀 Chạy thử

1. Cài đặt các thư viện cần thiết:
```bash
uv pip install -r requirements.txt
```

2. Chạy hệ thống:
```bash
python app.py
```
