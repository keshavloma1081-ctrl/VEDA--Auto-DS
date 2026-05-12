"""
Agent 92: Collaborative Filtering Agent
User-based and item-based collaborative filtering
"""
from typing import Dict, Any
import json
from .base_agent import RecommendationBaseAgent

class CollaborativeFilteringAgent(RecommendationBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations using collaborative filtering"""
        
        user_id = state.get('user_id', '')
        user_history = state.get('user_history', [])
        similarity_metric = state.get('similarity_metric', 'cosine')
        top_n = state.get('top_n', 10)
        
        prompt = f"""You are a collaborative filtering recommendation expert.

USER ID: {user_id}
USER HISTORY: {user_history}
SIMILARITY METRIC: {similarity_metric}
TOP N RECOMMENDATIONS: {top_n}

Generate recommendations based on similar users' preferences.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "recommendations": [
        {{
            "item_id": "item_123",
            "item_name": "Product A",
            "predicted_rating": 4.5,
            "confidence": 0.87,
            "reason": "Users similar to you loved this"
        }},
        {{
            "item_id": "item_456",
            "item_name": "Product B",
            "predicted_rating": 4.3,
            "confidence": 0.82,
            "reason": "Popular among similar users"
        }}
    ],
    "similar_users": ["user_789", "user_234"],
    "algorithm": "user_based|item_based",
    "similarity_metric": "{similarity_metric}",
    "coverage": 0.75,
    "diversity_score": 0.68
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
                "collaborative_filtering": result,
                "recommendations": result.get("recommendations", []),
                "algorithm": result.get("algorithm", "unknown"),
                "diversity_score": result.get("diversity_score", 0)
            }
        except Exception as e:
            return {
                "collaborative_filtering": {"error": f"Failed CF: {str(e)}"},
                "recommendations": [],
                "algorithm": "unknown"
            }