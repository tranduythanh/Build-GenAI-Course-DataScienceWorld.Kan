from typing import Union, Dict, Any, Optional
from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput
from datetime import datetime
from llama_data_collector import DataCollector
from logger import tool_logger
from collections.abc import Mapping
import os
from openai import OpenAI


class StockPriceTool(BaseTool):
    """Tool for getting stock price data"""
    
    def __init__(self) -> None:
        self.data_collector: DataCollector = DataCollector()
        # Initialize OpenAI client for LLM analysis
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm_client = OpenAI(api_key=api_key) if api_key else None
    
    def _get_llm_analysis(self, symbol: str, data_summary: str) -> str:
        """Get LLM analysis of stock data"""
        if not self.llm_client:
            return "🤖 **AI Analysis**: LLM not available (API key not configured)"
        
        try:
            prompt = f"""Bạn là một chuyên gia phân tích chứng khoán Việt Nam. Hãy phân tích dữ liệu cổ phiếu {symbol} sau đây và đưa ra nhận định tổng quan:

{data_summary}

Hãy cung cấp:
1. Nhận định về xu hướng giá
2. Điểm đáng chú ý trong dữ liệu
3. Khuyến nghị đầu tư (chỉ mang tính tham khảo)
4. Rủi ro cần lưu ý

Trả lời bằng tiếng Việt, ngắn gọn và dễ hiểu (tối đa 200 từ)."""

            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            
            return f"🤖 **AI Analysis**:\n{response.choices[0].message.content}"
        except Exception as e:
            tool_logger.warning(f"LLM analysis failed: {str(e)}")
            return f"🤖 **AI Analysis**: Unable to generate analysis ({str(e)})"
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="get_stock_price",
            description="""Get stock price data for Vietnamese stocks. 
            
            Input format options:
            1. Just symbol: "HAG" 
            2. Symbol with dates: {"symbol": "HAG", "start_date": "2024-01-01", "end_date": "2024-06-30"}
            3. Input wrapper: {"input": "HAG", "start_date": "2024-01-01", "end_date": "2024-06-30"}
            
            Supported symbols: HAG, VIC, FPT, VNM, MSN, VNINDEX
            Date format: YYYY-MM-DD
            Default date range: 2023-01-01 to today"""
        )
    
    def __call__(self, input: Union[str, Dict[str, Any]], **kwargs) -> ToolOutput:
        """Get stock price data for a given symbol and date range
        
        Args:
            input: Can be one of the following formats:
                - String: Just the symbol (e.g., "HAG", "VIC", "VNINDEX")
                - Dict with direct parameters: {"symbol": "HAG", "start_date": "2023-01-01", "end_date": "2024-01-01"}
                - Dict with wrapped input: {"input": "HAG", "start_date": "2023-01-01", "end_date": "2024-01-01"}
        
        Examples:
            tool("HAG")  # Get HAG data from 2023-01-01 to today
            tool({"symbol": "VIC", "start_date": "2023-06-01", "end_date": "2023-12-31"})
            tool({"input": "VNINDEX", "start_date": "2024-01-01"})  # end_date defaults to today
            
        Returns:
            ToolOutput with stock price data in raw_output as list of records
        """
        tool_name: str = self.metadata.name or "get_stock_price"
        
        try:
            # Normalize input to always be a dictionary for raw_input
            raw_input_dict: Dict[str, Any]
            symbol: Optional[str]
            start_date: Optional[str]
            end_date: Optional[str]
            
            if isinstance(input, str):
                # If input is just a string, treat it as the symbol
                symbol = input
                start_date = kwargs.get("start_date", "2023-01-01")
                end_date = kwargs.get("end_date", datetime.now().strftime("%Y-%m-%d"))
                raw_input_dict = {"symbol": symbol, "start_date": start_date, "end_date": end_date}
            elif isinstance(input, (dict, Mapping)):
                # Convert any mapping (including AttributedDict) to regular dict
                raw_input_dict = dict(input)
                # Add kwargs to raw_input_dict for logging
                raw_input_dict.update(kwargs)
                
                # Handle wrapped input format from LlamaIndex agent
                if "input" in raw_input_dict:
                    inner_input = raw_input_dict["input"]
                    if isinstance(inner_input, str):
                        # If input is just a string (symbol), use it
                        symbol = inner_input
                        start_date = raw_input_dict.get("start_date") or kwargs.get("start_date", "2023-01-01")
                        end_date = raw_input_dict.get("end_date") or kwargs.get("end_date", datetime.now().strftime("%Y-%m-%d"))
                    elif isinstance(inner_input, (dict, Mapping)):
                        # If input is a mapping, extract from it
                        inner_dict = dict(inner_input)
                        symbol = inner_dict.get("symbol")
                        start_date = inner_dict.get("start_date") or raw_input_dict.get("start_date") or kwargs.get("start_date", "2023-01-01")
                        end_date = inner_dict.get("end_date") or raw_input_dict.get("end_date") or kwargs.get("end_date", datetime.now().strftime("%Y-%m-%d"))
                    else:
                        symbol = raw_input_dict.get("symbol")
                        start_date = raw_input_dict.get("start_date") or kwargs.get("start_date")
                        end_date = raw_input_dict.get("end_date") or kwargs.get("end_date")
                else:
                    # Handle direct parameter format
                    symbol = raw_input_dict.get("symbol")
                    start_date = raw_input_dict.get("start_date") or kwargs.get("start_date")
                    end_date = raw_input_dict.get("end_date") or kwargs.get("end_date")
            else:
                return ToolOutput(
                    content="Invalid input format",
                    tool_name=tool_name,
                    raw_input={"error": "Invalid input format"},
                    raw_output={"error": "Invalid input format"},
                    is_error=True
                )
            
            if not self.data_collector:
                return ToolOutput(
                    content="API key not configured",
                    tool_name=tool_name,
                    raw_input=raw_input_dict,
                    raw_output={"error": "API key not configured"},
                    is_error=True
                )
            
            if not symbol:
                return ToolOutput(
                    content="Missing required parameter: symbol",
                    tool_name=tool_name,
                    raw_input=raw_input_dict,
                    raw_output={"error": "Missing required parameter: symbol"},
                    is_error=True
                )
            
            # Provide defaults if dates are missing
            if not start_date:
                start_date = "2023-01-01"
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
                
            tool_logger.info(f"Fetching stock data for {symbol} from {start_date} to {end_date}")
            df = self.data_collector.get_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            result = df.to_dict(orient="records")
            
            tool_logger.info(f"Successfully retrieved stock data for {symbol} from {start_date} to {end_date}")
            tool_logger.info(f"Number of rows: {len(result)}")
            if len(result) > 5:
                tool_logger.info(f"last 5 rows: {result[-5:]}")
            else:
                tool_logger.info(f"all rows: {result}")
            
            # Create detailed content message
            content_lines = []
            content_lines.append(f"📊 **Stock Data for {symbol}** ({start_date} to {end_date})")
            content_lines.append(f"📈 **Total data points**: {len(result)}")
            
            if result:
                # Get latest data (most recent)
                latest = result[-1]
                content_lines.append(f"\n**📅 Latest Trading Day:**")
                
                # Handle multi-level column format from vnquant
                close_price = latest.get('close', latest.get('adjust', latest.get(('close', symbol), latest.get(('adjust', symbol), 'N/A'))))
                high_price = latest.get('high', latest.get(('high', symbol), 'N/A'))
                low_price = latest.get('low', latest.get(('low', symbol), 'N/A'))
                volume = latest.get('volume_match', latest.get(('volume_match', symbol), 'N/A'))
                
                content_lines.append(f"- **Close**: {close_price} VND")
                content_lines.append(f"- **High**: {high_price} VND")
                content_lines.append(f"- **Low**: {low_price} VND")
                content_lines.append(f"- **Volume**: {volume}")
                
                # Show last 10 data points
                display_count = min(10, len(result))
                content_lines.append(f"\n**📋 Recent {display_count} Trading Days:**")
                content_lines.append("| Date | Close | High | Low | Volume |")
                content_lines.append("|------|-------|------|-----|--------|")
                
                for i, row in enumerate(result[-display_count:]):
                    # Handle multi-level columns
                    close_price = row.get('close', row.get('adjust', row.get(('close', symbol), row.get(('adjust', symbol), 'N/A'))))
                    high_price = row.get('high', row.get(('high', symbol), 'N/A'))
                    low_price = row.get('low', row.get(('low', symbol), 'N/A'))
                    volume = row.get('volume_match', row.get(('volume_match', symbol), 'N/A'))
                    
                    # Try to format volume nicely
                    if volume != 'N/A' and volume:
                        try:
                            vol_num = float(str(volume).replace(',', ''))
                            if vol_num >= 1_000_000:
                                volume = f"{vol_num/1_000_000:.1f}M"
                            elif vol_num >= 1_000:
                                volume = f"{vol_num/1_000:.1f}K"
                        except:
                            pass
                    
                    content_lines.append(f"| Day {len(result)-display_count+i+1} | {close_price} | {high_price} | {low_price} | {volume} |")
                
                # Calculate some basic stats
                try:
                    prices = []
                    for row in result:
                        # Handle multi-level columns
                        price = row.get('close', row.get('adjust', row.get(('close', symbol), row.get(('adjust', symbol)))))
                        if price and price != 'N/A':
                            prices.append(float(str(price).replace(',', '')))
                    
                    if len(prices) >= 2:
                        price_change = prices[-1] - prices[0]
                        price_change_pct = (price_change / prices[0]) * 100
                        content_lines.append(f"\n**📊 Period Performance:**")
                        content_lines.append(f"- **Change**: {price_change:+.2f} VND ({price_change_pct:+.2f}%)")
                        content_lines.append(f"- **Highest**: {max(prices):.2f} VND")
                        content_lines.append(f"- **Lowest**: {min(prices):.2f} VND")
                except Exception as e:
                    content_lines.append(f"\n**Note**: Could not calculate performance metrics")
            
            # Add LLM analysis
            try:
                # Create summary for LLM
                llm_summary = f"""
Symbol: {symbol}
Period: {start_date} to {end_date}
Total data points: {len(result)}
Latest price: {close_price} VND
Price range: {min(prices):.2f} - {max(prices):.2f} VND
Performance: {price_change:+.2f} VND ({price_change_pct:+.2f}%)
Volume trend: Last trading volume was {volume}
                """.strip()
                
                llm_analysis = self._get_llm_analysis(symbol, llm_summary)
                content_lines.append(f"\n{llm_analysis}")
            except Exception as e:
                content_lines.append(f"\n🤖 **AI Analysis**: Analysis unavailable")
                    
            content_message = "\n".join(content_lines)
            
            return ToolOutput(
                content=content_message,
                tool_name=tool_name,
                raw_input=raw_input_dict,
                raw_output=result
            )
        except Exception as e:
            tool_logger.error(f"Error in get_stock_price: {str(e)}")
            return ToolOutput(
                content=f"Error getting stock price: {str(e)}",
                tool_name=tool_name,
                raw_input=raw_input_dict if 'raw_input_dict' in locals() else {"error": "Failed to process input"},
                raw_output={"error": str(e)},
                is_error=True
            ) 
        
if __name__ == "__main__":
    tool: StockPriceTool = StockPriceTool()
    print("\n\n\nmetadata")
    print(tool.metadata)
    
    # Ví dụ 1: Chỉ symbol (sử dụng default date range)
    print("\n\n\n=== Ví dụ 1: Chỉ symbol (HAG) ===")
    print(tool("HAG"))
    
    print("\n\n\n=== Ví dụ 2: Chỉ symbol (VIC) ===")
    print(tool("VIC"))
    
    print("\n\n\n=== Ví dụ 3: Chỉ symbol (VNINDEX) ===")
    print(tool("VNINDEX"))
    
    # Ví dụ 4: Dictionary format với đầy đủ tham số
    print("\n\n\n=== Ví dụ 4: Dictionary format với đầy đủ tham số ===")
    print(tool({
        "symbol": "FPT", 
        "start_date": "2024-01-01", 
        "end_date": "2024-06-30"
    }))
    
    # Ví dụ 5: Dictionary format với chỉ start_date
    print("\n\n\n=== Ví dụ 5: Dictionary format với chỉ start_date ===")
    print(tool({
        "symbol": "MSN", 
        "start_date": "2024-06-01"
    }))
    
    # Ví dụ 6: Wrapped input format (như từ LlamaIndex agent)
    print("\n\n\n=== Ví dụ 6: Wrapped input format ===")
    print(tool({
        "input": "TCB",
        "start_date": "2024-01-01",
        "end_date": "2024-03-31"
    }))
    
    # Ví dụ 7: Wrapped input format chỉ với symbol
    print("\n\n\n=== Ví dụ 7: Wrapped input format chỉ với symbol ===")
    print(tool({
        "input": "BID"
    }))
    
    # Ví dụ 8: Test với symbol khác
    print("\n\n\n=== Ví dụ 8: Symbol khác (CTG) ===")
    print(tool({
        "symbol": "CTG",
        "start_date": "2023-12-01",
        "end_date": "2023-12-31"
    }))
    