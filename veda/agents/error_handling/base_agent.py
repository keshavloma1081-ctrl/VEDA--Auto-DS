"""
Base Agent for Error Handling with Groq Support
"""
from groq import Groq
import os

class ErrorHandlingBaseAgent:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
    
    def call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call Groq API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1
        )
        return response.choices[0].message.content