"""
Agent 24: Question Answering Agent
Answers questions based on provided context
"""
from typing import Dict, Any
import json
from groq import Groq
import os

class QuestionAnsweringAgent:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Answer question based on context"""
        
        context = state.get('context', '')
        question = state.get('question', '')
        
        prompt = f"""Answer the question based ONLY on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

Return ONLY valid JSON (no markdown, no backticks):
{{
    "answer": "the answer extracted from context",
    "confidence": 0.92,
    "evidence": "relevant excerpt from context supporting the answer",
    "answer_type": "factual|opinion|yes_no|numeric",
    "source_sentence": "exact sentence containing the answer"
}}

If the answer is not in the context, set answer to "Answer not found in context".
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
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
                "question_answering": result,
                "answer": result.get("answer", ""),
                "confidence": result.get("confidence", 0),
                "answer_type": result.get("answer_type", "unknown")
            }
        except Exception as e:
            return {
                "question_answering": {"error": f"Failed to answer question: {str(e)}"},
                "answer": "",
                "confidence": 0
            }
    