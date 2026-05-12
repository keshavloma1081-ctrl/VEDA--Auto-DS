"""
Agent 95: Cold Start Handler Agent
Handles new users and new items with limited data
"""
from typing import Dict, Any
import json
from .base_agent import RecommendationBaseAgent

class ColdStartHandlerAgent(RecommendationBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cold start problem"""
        
        entity_type = state.get('entity_type', 'user')  # 'user' or 'item'
        entity_id = state.get('entity_id', '')
        available_data = state.get('available_data', {})
        strategy = state.get('strategy', 'popularity')
        
        prompt = f"""You are a cold start problem expert.

ENTITY TYPE: {entity_type}
ENTITY ID: {entity_id}
AVAILABLE DATA: {available_data}
STRATEGY: {strategy}

Handle cold start problem with limited data.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "recommendations": [
        {{
            "item_id": "item_popular_1",
            "item_name": "Trending Product",
            "score": 0.85,
            "strategy_used": "popularity|demographic|content_features",
            "reason": "Popular among all users"
        }}
    ],
    "cold_start_strategy": "popularity_based|demographic_filtering|exploration|onboarding_questions",
    "confidence_level": "low|medium|high",
    "exploration_rate": 0.3,
    "data_collection_suggestions": [
        "Ask user preferences",
        "Show diverse sample items",
        "Collect explicit ratings"
    ],
    "expected_improvement_after_interactions": 5
}}
"""
        
        try:
            response_text = self.call_llm(prompt, max_tokens=1500).strip()
            
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            return {
                "cold_start_handler": result,
                "recommendations": result.get("recommendations", []),
                "strategy": result.get("cold_start_strategy", "unknown"),
                "confidence_level": result.get("confidence_level", "low")
            }
        except Exception as e:
            return {
                "cold_start_handler": {"error": f"Failed cold start: {str(e)}"},
                "recommendations": [],
                "strategy": "unknown"
            }