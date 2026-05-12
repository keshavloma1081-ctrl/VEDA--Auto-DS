"""
Agent 11: PDF Processor Agent
Extracts text, tables, and metadata from PDF documents
"""
from typing import Dict, Any
import json
from .base_agent import BaseAgent

class PDFProcessorAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data from PDFs"""
        
        pdf_path = state.get('pdf_path', 'test_data/data_sources/pdfs/quarterly_report.txt')
        try:
            with open(pdf_path, 'r', encoding='utf-8') as f:
                pdf_content = f.read()
        except:
            pdf_content = "No PDF content available"
        
        prompt = f"""You are a PDF data extraction specialist.

PDF CONTENT:
{pdf_content}

EXTRACTION TASK: {state.get('extraction_task', 'Extract all text and tables')}

Your tasks:
1. Identify document structure (headers, sections, tables)
2. Extract text with layout preservation
3. Detect and extract tables
4. Parse metadata (author, creation date, page count)

Return ONLY valid JSON (no markdown, no backticks):
{{
    "document_type": "report",
    "page_count": 1,
    "extracted_text": "summary of content",
    "tables": [{{"table_id": 1, "headers": ["col1"], "rows": [["val1"]]}}],
    "metadata": {{"author": "name", "created": "date", "title": "title"}},
    "sections": [{{"heading": "section", "content": "content", "page": 1}}],
    "processing_notes": "any issues"
}}
"""
        
        try:
            response_text = self.call_llm(prompt, max_tokens=4000).strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            return {
                "pdf_processor": result,
                "document_type": result.get("document_type", "unknown"),
                "tables_extracted": len(result.get("tables", [])),
                "pages_processed": result.get("page_count", 0),
                "has_tables": len(result.get("tables", [])) > 0
            }
        except Exception as e:
            return {
                "pdf_processor": {"error": f"Failed to parse PDF: {str(e)}"},
                "document_type": "unknown",
                "tables_extracted": 0,
                "pages_processed": 0
            }