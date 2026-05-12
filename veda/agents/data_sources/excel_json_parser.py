"""
Agent 14: Excel/JSON Parser Agent
Parses structured files into normalized data
"""
from typing import Dict, Any
import json
from .base_agent import BaseAgent

class ExcelJSONParserAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Excel and JSON files intelligently"""
        
        file_type = state.get('file_type', 'csv')
        file_path = state.get('file_path', 'test_data/data_sources/files/products_inventory.csv')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except:
            file_content = "No file content available"
        
        prompt = f"""You are a structured data parsing expert.

FILE TYPE: {file_type}
FILE CONTENT:
{file_content}

Your tasks:
1. Identify data structure and headers
2. Detect data types for each column
3. Find and handle missing values
4. Validate data quality

Return ONLY valid JSON (no markdown, no backticks):
{{
    "file_format": "csv",
    "sheets_found": ["Sheet1"],
    "total_rows": 10,
    "columns": [
        {{"name": "column_name", "type": "string", "null_count": 0}}
    ],
    "data_quality_score": 95,
    "parsing_notes": "issues found",
    "normalized_schema": {{}},
    "sample_rows": [[]]
}}
"""
        
        try:
            response_text = self.call_llm(prompt, max_tokens=3000).strip()
            
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            return {
                "excel_json_parser": result,
                "file_format": result.get("file_format", "unknown"),
                "total_rows": result.get("total_rows", 0),
                "data_quality_score": result.get("data_quality_score", 0)
            }
        except Exception as e:
            return {
                "excel_json_parser": {"error": f"Failed to parse file: {str(e)}"},
                "file_format": "unknown"
            }