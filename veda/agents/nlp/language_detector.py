"""
Agent 22: Language Detector Agent
Identifies the language of input text
"""
from typing import Dict, Any
import json
from groq import Groq
import os

class LanguageDetectorAgent:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect language of text"""
        
        text = state.get('text', '')
        
        prompt = f"""Detect the language of this text.

TEXT:
{text}

Return ONLY valid JSON (no markdown, no backticks):
{{
    "language": "English",
    "language_code": "en",
    "confidence": 0.98,
    "script": "Latin",
    "alternative_languages": [{{"language": "Spanish", "confidence": 0.02}}]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            return {
                "language_detector": result,
                "language": result.get("language", "unknown"),
                "language_code": result.get("language_code", "unknown"),
                "confidence": result.get("confidence", 0)
            }
        except Exception as e:
            return {
                "language_detector": {"error": f"Failed to detect language: {str(e)}"},
                "language": "unknown",
                "confidence": 0
            }