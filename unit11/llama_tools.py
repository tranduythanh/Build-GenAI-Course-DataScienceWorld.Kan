from llama_tool_stock_price import StockPriceTool
from llama_tool_technical_analysis import TechnicalAnalysisTool
from llama_tool_web_data import WebDataTool
from llama_tool_finance_report import FinanceReportTool


# Factory function for creating tools
def create_stock_tools(api_key: str = None):
    """Create and return a list of stock analysis tools"""
    return [
        StockPriceTool(api_key=api_key),
        TechnicalAnalysisTool(),
        WebDataTool(),
        FinanceReportTool()
    ] 