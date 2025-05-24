from typing import Dict, Any
from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput
from datetime import datetime


class FinanceReportTool(BaseTool):
    """Tool for generating financial analysis reports"""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="generate_finance_report",
            description="Generate financial analysis reports. Requires data and report_type ('technical' or 'fundamental')."
        )
    
    def __call__(self, input: Dict[str, Any]) -> ToolOutput:
        """Generate financial analysis reports"""
        try:
            # Handle wrapped input format from LlamaIndex agent
            if "input" in input:
                if isinstance(input["input"], (dict, list)):
                    data = input["input"]
                    report_type = input.get("report_type", "technical")
                else:
                    # If input is just a string, try to get data from the main input
                    data = input.get("data")
                    report_type = input.get("report_type", "technical")
            else:
                data = input.get("data")
                report_type = input.get("report_type", "technical")
            
            if not data:
                return ToolOutput(
                    content="Missing required parameter: data",
                    tool_name=self.metadata.name,
                    raw_input=input,
                    raw_output={"error": "Missing required parameter: data"},
                    is_error=True
                )
            
            if report_type == "technical":
                result = {
                    "type": "technical",
                    "summary": "Technical Analysis Report",
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
            elif report_type == "fundamental":
                result = {
                    "type": "fundamental",
                    "summary": "Fundamental Analysis Report",
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return ToolOutput(
                    content="Unsupported report type. Use 'technical' or 'fundamental'",
                    tool_name=self.metadata.name,
                    raw_input=input,
                    raw_output={"error": "Unsupported report type"},
                    is_error=True
                )
            
            return ToolOutput(
                content=f"Successfully generated {report_type} analysis report",
                tool_name=self.metadata.name,
                raw_input=input,
                raw_output=result
            )
        except Exception as e:
            return ToolOutput(
                content=f"Error generating report: {str(e)}",
                tool_name=self.metadata.name,
                raw_input=input,
                raw_output={"error": str(e)},
                is_error=True
            ) 