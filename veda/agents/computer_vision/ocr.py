"""
Agent 88: OCR Agent
Extracts text from images
"""
from typing import Dict, Any
import json
from .base_agent import CVBaseAgent

class OCRAgent(CVBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from image"""
        
        image_path = state.get('image_path', '')
        languages = state.get('languages', ['en'])
        
        prompt = f"""You are an OCR expert.

IMAGE PATH: {image_path}
LANGUAGES: {languages}

Extract all text from the image.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "extracted_text": "The complete text extracted from the image",
    "text_blocks": [
        {{
            "text": "Block 1 text",
            "bbox": {{"x": 50, "y": 100, "width": 300, "height": 50}},
            "confidence": 0.96,
            "language": "en"
        }}
    ],
    "word_count": 25,
    "average_confidence": 0.94,
    "ocr_engine": "tesseract|easyocr|paddleocr",
    "processing_time_ms": 150,
    "text_orientation": "horizontal|vertical"
}}
"""
        
        try:
            response_text = self.call_llm(prompt, max_tokens=2000).strip()
            
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            return {
                "ocr": result,
                "extracted_text": result.get("extracted_text", ""),
                "text_blocks": result.get("text_blocks", []),
                "word_count": result.get("word_count", 0)
            }
        except Exception as e:
            return {
                "ocr": {"error": f"Failed OCR: {str(e)}"},
                "extracted_text": "",
                "text_blocks": []
            }