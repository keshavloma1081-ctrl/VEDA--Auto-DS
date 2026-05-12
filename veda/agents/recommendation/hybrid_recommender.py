"""
Agent 94: Hybrid Recommender Agent
Combines multiple recommendation strategies
"""
from typing import Dict, Any
import json
from .base_agent import RecommendationBaseAgent

class HybridRecommenderAgent(RecommendationBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hybrid recommendations"""
        
        user_id = state.get('user_id', '')
        cf_results = state.get('cf_results', [])
        cb_results = state.get('cb_results', [])
        weights = state.get('weights', {'cf': 0.6, 'cb': 0.4})
        top_n = state.get('top_n', 10)
        
        prompt = f"""You are a hybrid recommendation expert.

USER ID: {user_id}
COLLABORATIVE FILTERING RESULTS: {len(cf_results)} items
CONTENT-BASED RESULTS: {len(cb_results)} items
WEIGHTS: CF={weights.get('cf')}, CB={weights.get('cb')}
TOP N RECOMMENDATIONS: {top_n}

Combine multiple recommendation strategies optimally.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "recommendations": [
        {{
            "item_id": "item_999",
            "item_name": "Product E",
            "hybrid_score": 0.91,
            "cf_score": 0.88,
            "cb_score": 0.95,
            "source": "both|cf_only|cb_only",
            "reason": "Top pick from both strategies"
        }}
    ],
    "ensemble_method": "weighted_average|rank_fusion|stacking",
    "cf_weight": {weights.get('cf')},
    "cb_weight": {weights.get('cb')},
    "diversity_score": 0.72,
    "coverage": 0.68,
    "performance_metrics": {{
        "precision": 0.85,
        "recall": 0.78,
        "ndcg": 0.82
    }}
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
                "hybrid_recommender": result,
                "recommendations": result.get("recommendations", []),
                "ensemble_method": result.get("ensemble_method", "unknown"),
                "diversity_score": result.get("diversity_score", 0)
            }
        except Exception as e:
            return {
                "hybrid_recommender": {"error": f"Failed hybrid: {str(e)}"},
                "recommendations": [],
                "ensemble_method": "unknown"
            }