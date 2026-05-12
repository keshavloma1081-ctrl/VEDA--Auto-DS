"""
Agent 25: Text Generator Agent
Generates new text based on prompts and parameters
"""
from typing import Dict, Any
import json
from groq import Groq
import os

class TextGeneratorAgent:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text based on prompt"""
        
        prompt_text = state.get('prompt', '')
        style = state.get('style', 'neutral')
        max_length = state.get('max_length', 200)
        temperature = state.get('temperature', 0.7)
        
        prompt = f"""Generate text based on this prompt.

PROMPT:
{prompt_text}

STYLE: {style}
MAX LENGTH: {max_length} words

Return ONLY valid JSON (no markdown, no backticks):
{{
    "generated_text": "the generated text here",
    "word_count": 150,
    "style_applied": "{style}",
    "creativity_level": "low|medium|high",
    "generation_method": "completion|summarization|expansion"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=temperature
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            return {
                "text_generator": result,
                "generated_text": result.get("generated_text", ""),
                "word_count": result.get("word_count", 0),
                "style_applied": result.get("style_applied", style)
            }
        except Exception as e:
            return {
                "text_generator": {"error": f"Failed to generate text: {str(e)}"},
                "generated_text": "",
                "word_count": 0
            }