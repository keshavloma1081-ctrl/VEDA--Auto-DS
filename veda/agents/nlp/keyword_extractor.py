"""
Agent 21: Keyword Extractor Agent
Extracts key terms and phrases from text
"""
from typing import Dict, Any
import json
from groq import Groq
import os

class KeywordExtractorAgent:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract keywords from text"""
        
        text = state.get('text', '')
        num_keywords = state.get('num_keywords', 10)
        
        prompt = f"""Extract the most important keywords and key phrases from this text.

TEXT:
{text}

Return ONLY valid JSON (no markdown, no backticks):
{{
    "keywords": ["keyword1", "keyword2", ...],
    "key_phrases": ["phrase1", "phrase2", ...],
    "topics": ["topic1", "topic2"],
    "importance_scores": {{"keyword1": 0.95, "keyword2": 0.87}},
    "extraction_method": "tfidf|llm|hybrid"
}}

Extract {num_keywords} top keywords.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
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
                "keyword_extractor": result,
                "keywords": result.get("keywords", []),
                "key_phrases": result.get("key_phrases", []),
                "topics": result.get("topics", [])
            }
        except Exception as e:
            return {
                "keyword_extractor": {"error": f"Failed to extract keywords: {str(e)}"},
                "keywords": [],
                "key_phrases": []
            }