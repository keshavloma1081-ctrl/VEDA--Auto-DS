"""
Agent 93: Content-Based Filtering Agent
Recommendations based on item features and user preferences
"""
from typing import Dict, Any
import json
from .base_agent import RecommendationBaseAgent

class ContentBasedAgent(RecommendationBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content-based recommendations"""
        
        user_id = state.get('user_id', '')
        user_profile = state.get('user_profile', {})
        item_catalog = state.get('item_catalog', [])
        top_n = state.get('top_n', 10)
        
        prompt = f"""You are a content-based recommendation expert.

USER ID: {user_id}
USER PROFILE: {user_profile}
ITEM CATALOG SIZE: {len(item_catalog)}
TOP N RECOMMENDATIONS: {top_n}

Generate recommendations based on item features matching user preferences.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "recommendations": [
        {{
            "item_id": "item_789",
            "item_name": "Product C",
            "match_score": 0.92,
            "matched_features": ["genre_action", "rating_high", "year_2023"],
            "reason": "Matches your preference for action content"
        }},
        {{
            "item_id": "item_012",
            "item_name": "Product D",
            "match_score": 0.88,
            "matched_features": ["genre_adventure", "rating_high"],
            "reason": "Similar to items you've enjoyed"
        }}
    ],
    "user_preference_vector": {{"action": 0.9, "adventure": 0.7, "comedy": 0.3}},
    "feature_weights": {{"genre": 0.5, "rating": 0.3, "year": 0.2}},
    "algorithm": "tfidf|cosine_similarity|neural_embedding",
    "novelty_score": 0.65
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
                "content_based": result,
                "recommendations": result.get("recommendations", []),
                "algorithm": result.get("algorithm", "unknown"),
                "novelty_score": result.get("novelty_score", 0)
            }
        except Exception as e:
            return {
                "content_based": {"error": f"Failed content-based: {str(e)}"},
                "recommendations": [],
                "algorithm": "unknown"
            }