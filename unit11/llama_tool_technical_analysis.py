from typing import Dict, Any, Union, List
from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput
import pandas as pd
import pandas_ta as ta
from llama_data_collector import DataCollector
from datetime import datetime
from collections.abc import Mapping
import os
from openai import OpenAI


class TechnicalAnalysisTool(BaseTool):
    """Tool for calculating technical indicators"""
    
    def __init__(self) -> None:
        self.data_collector: DataCollector = DataCollector()
        # Initialize OpenAI client for LLM analysis
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm_client = OpenAI(api_key=api_key) if api_key else None
    
    def _get_llm_technical_analysis(self, symbol: str, technical_summary: str) -> str:
        """Get LLM analysis of technical indicators"""
        if not self.llm_client:
            return "🤖 **AI Technical Analysis**: LLM not available (API key not configured)"
        
        try:
            prompt = f"""Bạn là một chuyên gia phân tích kỹ thuật chứng khoán Việt Nam. Hãy phân tích các chỉ số kỹ thuật của cổ phiếu {symbol} sau đây:

{technical_summary}

Hãy cung cấp:
1. Tổng quan xu hướng dựa trên các chỉ số kỹ thuật
2. Tín hiệu mua/bán từ các indicators
3. Mức hỗ trợ/kháng cự quan trọng
4. Khuyến nghị giao dịch ngắn hạn (chỉ mang tính tham khảo)

Trả lời bằng tiếng Việt, ngắn gọn và chuyên nghiệp (tối đa 250 từ)."""

            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.7
            )
            
            return f"🤖 **AI Technical Analysis**:\n{response.choices[0].message.content}"
        except Exception as e:
            return f"🤖 **AI Technical Analysis**: Unable to generate analysis ({str(e)})"
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="calculate_technical_indicators",
            description="""Calculate technical indicators for Vietnamese stocks.
            
            Input options:
            1. Stock symbol: "HAG" (will get stock data first, then calculate indicators)
            2. Raw stock data: Pass stock data directly from get_stock_price tool
            
            Calculates: SMA_20, RSI_14, MACD, Bollinger Bands
            Works with: HAG, VIC, FPT, VNM, MSN, VNINDEX"""
        )
    
    def __call__(self, input: Union[str, Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> ToolOutput:
        """Calculate technical indicators for stock data"""
        try:
            # Normalize input to always be a dictionary for raw_input
            data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame, None] = None
            raw_input_dict: Dict[str, Any]
            symbol: str = ""
            
            # Handle different input formats
            if isinstance(input, str):
                # If input is just a string (symbol), get stock data first
                symbol = input
                raw_input_dict = {"symbol": symbol}
                # Get stock data
                df_stock = self.data_collector.get_stock_data(
                    symbol=symbol,
                    start_date="2023-01-01",
                    end_date=datetime.now().strftime("%Y-%m-%d")
                )
                data = df_stock.to_dict(orient="records")
            elif isinstance(input, (dict, Mapping)):
                raw_input_dict = dict(input)
                # Handle wrapped input format from LlamaIndex agent
                if "input" in raw_input_dict:
                    inner_input = raw_input_dict["input"]
                    if isinstance(inner_input, str):
                        # If input is just a string (symbol), get stock data first
                        symbol = inner_input
                        df_stock = self.data_collector.get_stock_data(
                            symbol=symbol,
                            start_date="2023-01-01",
                            end_date=datetime.now().strftime("%Y-%m-%d")
                        )
                        data = df_stock.to_dict(orient="records")
                    elif isinstance(inner_input, (dict, list, Mapping)):
                        if isinstance(inner_input, Mapping):
                            data = dict(inner_input)
                        else:
                            data = inner_input
                    else:
                        data = raw_input_dict.get("data")
                else:
                    data = raw_input_dict.get("data")
                    if not data and "symbol" in raw_input_dict:
                        symbol = raw_input_dict["symbol"]
                        df_stock = self.data_collector.get_stock_data(
                            symbol=symbol,
                            start_date="2023-01-01",
                            end_date=datetime.now().strftime("%Y-%m-%d")
                        )
                        data = df_stock.to_dict(orient="records")
            elif isinstance(input, list):
                data = input
                raw_input_dict = {"data": data}
            else:
                raw_input_dict = {"error": "Invalid input format"}
                data = None
                
            if not data:
                return ToolOutput(
                    content="Could not get stock data. Please provide symbol or stock data.",
                    tool_name=self.metadata.name,
                    raw_input=raw_input_dict,
                    raw_output={"error": "Could not get stock data"},
                    is_error=True
                )
            
            # Convert dict back to DataFrame if needed
            df: pd.DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
                
            # Ensure required columns exist and convert to numeric
            if 'close' not in df.columns:
                if 'Close' in df.columns:
                    df['close'] = pd.to_numeric(df['Close'], errors='coerce')
                elif 'adjust' in df.columns:
                    df['close'] = pd.to_numeric(df['adjust'], errors='coerce')
                else:
                    # Use the last numeric column as close price
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        df['close'] = df[numeric_cols[-1]]
                    else:
                        raise ValueError("No numeric columns found for close price")
            else:
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                
            # Remove rows with NaN close prices
            df = df.dropna(subset=['close'])
                
            # Calculate SMA
            df['SMA_20'] = ta.sma(df['close'], length=20)
            
            # Calculate RSI
            df['RSI_14'] = ta.rsi(df['close'], length=14)
            
            # Calculate MACD
            macd_data = ta.macd(df['close'])
            if macd_data is not None:
                df = pd.concat([df, macd_data], axis=1)
            
            # Calculate Bollinger Bands
            bb_data = ta.bbands(df['close'], length=20)
            if bb_data is not None:
                df = pd.concat([df, bb_data], axis=1)
            
            result: List[Dict[str, Any]] = df.to_dict(orient="records")
            
            # Create detailed content message with technical analysis
            content_lines = []
            if symbol:
                content_lines.append(f"📈 **Technical Analysis for {symbol}**")
            else:
                content_lines.append(f"📈 **Technical Analysis Results**")
            content_lines.append(f"📊 **Total data points**: {len(result)}")
            
            if result:
                # Get latest technical indicators
                latest = result[-1]
                content_lines.append(f"\n**📅 Latest Technical Indicators:**")
                
                # Current price
                close_price = latest.get('close', latest.get('adjust', 'N/A'))
                content_lines.append(f"- **Current Price**: {close_price} VND")
                
                # SMA
                sma_20 = latest.get('SMA_20')
                if sma_20 and not pd.isna(sma_20):
                    content_lines.append(f"- **SMA(20)**: {sma_20:.2f} VND")
                    try:
                        if float(close_price) > sma_20:
                            content_lines.append(f"  ↗️ Price above SMA(20) - **Bullish trend**")
                        else:
                            content_lines.append(f"  ↘️ Price below SMA(20) - **Bearish trend**")
                    except:
                        pass
                
                # RSI
                rsi_14 = latest.get('RSI_14')
                if rsi_14 and not pd.isna(rsi_14):
                    content_lines.append(f"- **RSI(14)**: {rsi_14:.2f}")
                    if rsi_14 > 70:
                        content_lines.append(f"  ⚠️ **Overbought** (RSI > 70)")
                    elif rsi_14 < 30:
                        content_lines.append(f"  ⚠️ **Oversold** (RSI < 30)")
                    else:
                        content_lines.append(f"  ✅ **Neutral** (30 < RSI < 70)")
                
                # MACD
                macd_line = latest.get('MACD_12_26_9')
                macd_signal = latest.get('MACDs_12_26_9')
                if macd_line and macd_signal and not pd.isna(macd_line) and not pd.isna(macd_signal):
                    content_lines.append(f"- **MACD**: {macd_line:.4f}")
                    content_lines.append(f"- **MACD Signal**: {macd_signal:.4f}")
                    if macd_line > macd_signal:
                        content_lines.append(f"  ↗️ **Bullish** (MACD > Signal)")
                    else:
                        content_lines.append(f"  ↘️ **Bearish** (MACD < Signal)")
                
                # Bollinger Bands
                bb_upper = latest.get('BBU_20_2.0')
                bb_lower = latest.get('BBL_20_2.0')
                bb_middle = latest.get('BBM_20_2.0')
                if bb_upper and bb_lower and not pd.isna(bb_upper) and not pd.isna(bb_lower):
                    content_lines.append(f"- **Bollinger Bands**:")
                    content_lines.append(f"  - Upper: {bb_upper:.2f} VND")
                    content_lines.append(f"  - Middle: {bb_middle:.2f} VND" if bb_middle and not pd.isna(bb_middle) else "")
                    content_lines.append(f"  - Lower: {bb_lower:.2f} VND")
                    
                    try:
                        price_val = float(close_price)
                        if price_val >= bb_upper:
                            content_lines.append(f"  ⚠️ **Price at/above upper band** - Potential resistance")
                        elif price_val <= bb_lower:
                            content_lines.append(f"  ⚠️ **Price at/below lower band** - Potential support")
                        else:
                            content_lines.append(f"  ✅ **Price within bands** - Normal range")
                    except:
                        pass
                
                # Show recent 5 days indicators summary
                display_count = min(5, len(result))
                content_lines.append(f"\n**📋 Recent {display_count} Days Summary:**")
                content_lines.append("| Day | Price | SMA(20) | RSI(14) | MACD Signal |")
                content_lines.append("|-----|-------|---------|---------|-------------|")
                
                for i, row in enumerate(result[-display_count:]):
                    day_num = len(result) - display_count + i + 1
                    price = row.get('close', row.get('adjust', 'N/A'))
                    sma = row.get('SMA_20', 'N/A')
                    rsi = row.get('RSI_14', 'N/A')
                    macd_sig = row.get('MACDs_12_26_9', 'N/A')
                    
                    # Format values
                    sma_str = f"{sma:.2f}" if sma != 'N/A' and not pd.isna(sma) else 'N/A'
                    rsi_str = f"{rsi:.1f}" if rsi != 'N/A' and not pd.isna(rsi) else 'N/A'
                    macd_str = f"{macd_sig:.3f}" if macd_sig != 'N/A' and not pd.isna(macd_sig) else 'N/A'
                    
                    content_lines.append(f"| {day_num} | {price} | {sma_str} | {rsi_str} | {macd_str} |")
                
                content_lines.append(f"\n**💡 Analysis Summary:**")
                content_lines.append(f"- Technical indicators calculated successfully")
                content_lines.append(f"- Use RSI for momentum analysis (overbought/oversold)")
                content_lines.append(f"- Use MACD for trend direction")
                content_lines.append(f"- Use Bollinger Bands for volatility and support/resistance")
                
                # Add LLM technical analysis
                try:
                    if symbol and latest:
                        # Safely extract values with defaults
                        current_price = str(close_price)
                        sma_val = sma_20 if sma_20 and not pd.isna(sma_20) else None
                        rsi_val = rsi_14 if rsi_14 and not pd.isna(rsi_14) else None
                        macd_val = macd_line if macd_line and not pd.isna(macd_line) else None
                        macd_sig_val = macd_signal if macd_signal and not pd.isna(macd_signal) else None
                        bb_u = bb_upper if bb_upper and not pd.isna(bb_upper) else None
                        bb_l = bb_lower if bb_lower and not pd.isna(bb_lower) else None
                        
                        # Create technical summary for LLM
                        sma_str = f"{sma_val:.2f}" if sma_val else 'N/A'
                        rsi_str = f"{rsi_val:.2f}" if rsi_val else 'N/A'
                        macd_str = f"{macd_val:.4f}" if macd_val else 'N/A'
                        macd_sig_str = f"{macd_sig_val:.4f}" if macd_sig_val else 'N/A'
                        bb_u_str = f"{bb_u:.2f}" if bb_u else 'N/A'
                        bb_l_str = f"{bb_l:.2f}" if bb_l else 'N/A'
                        
                        trend = "Bullish" if sma_val and float(current_price) > sma_val else "Bearish" if sma_val else "Unknown"
                        rsi_signal = "Overbought" if rsi_val and rsi_val > 70 else "Oversold" if rsi_val and rsi_val < 30 else "Neutral" if rsi_val else "Unknown"
                        macd_trend = "Bullish" if macd_val and macd_sig_val and macd_val > macd_sig_val else "Bearish" if macd_val and macd_sig_val else "Unknown"
                        
                        tech_summary = f"""
Symbol: {symbol}
Current Price: {current_price} VND
SMA(20): {sma_str} VND
Trend: {trend}
RSI(14): {rsi_str}
RSI Signal: {rsi_signal}
MACD: {macd_str}
MACD Signal: {macd_sig_str}
MACD Trend: {macd_trend}
Bollinger Upper: {bb_u_str} VND
Bollinger Lower: {bb_l_str} VND
                        """.strip()
                        
                        llm_analysis = self._get_llm_technical_analysis(symbol, tech_summary)
                        content_lines.append(f"\n{llm_analysis}")
                    else:
                        content_lines.append(f"\n🤖 **AI Technical Analysis**: Unable to generate summary")
                except Exception as e:
                    content_lines.append(f"\n🤖 **AI Technical Analysis**: Analysis error - {str(e)}")
                    
            content_message = "\n".join(content_lines)
            
            return ToolOutput(
                content=content_message,
                tool_name=self.metadata.name,
                raw_input=raw_input_dict,
                raw_output=result
            )
        except Exception as e:
            return ToolOutput(
                content=f"Error calculating technical indicators: {str(e)}",
                tool_name=self.metadata.name,
                raw_input=raw_input_dict if 'raw_input_dict' in locals() else {"error": "Failed to process input"},
                raw_output={"error": str(e)},
                is_error=True
            ) 