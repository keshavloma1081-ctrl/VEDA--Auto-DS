"""
Agent 23: Translation Agent
Translates text between languages
"""
from typing import Dict, Any
import json
from groq import Groq
import os

class TranslationAgent:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Translate text to target language"""
        
        text = state.get('text', '')
        target_language = state.get('target_language', 'Spanish')
        source_language = state.get('source_language', 'auto-detect')
        
        prompt = f"""Translate this text to {target_language}.

SOURCE TEXT ({source_language}):
{text}

Return ONLY valid JSON (no markdown, no backticks):
{{
    "translated_text": "translated version here",
    "source_language": "English",
    "target_language": "{target_language}",
    "confidence": 0.95,
    "translation_method": "neural|rule_based"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
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
                "translation": result,
                "translated_text": result.get("translated_text", ""),
                "source_language": result.get("source_language", "unknown"),
                "target_language": result.get("target_language", target_language)
            }
        except Exception as e:
            return {
                "translation": {"error": f"Failed to translate: {str(e)}"},
                "translated_text": "",
                "target_language": target_language
            }