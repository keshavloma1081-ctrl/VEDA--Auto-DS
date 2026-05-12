"""
Agent 96: Diversity Optimizer Agent
Optimizes recommendation diversity while maintaining relevance
"""
from typing import Dict, Any
import json
from .base_agent import RecommendationBaseAgent

class DiversityOptimizerAgent(RecommendationBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize recommendation diversity"""
        
        initial_recommendations = state.get('initial_recommendations', [])
        diversity_weight = state.get('diversity_weight', 0.5)
        relevance_weight = state.get('relevance_weight', 0.5)
        top_n = state.get('top_n', 10)
        
        prompt = f"""You are a recommendation diversity expert.

INITIAL RECOMMENDATIONS: {len(initial_recommendations)} items
DIVERSITY WEIGHT: {diversity_weight}
RELEVANCE WEIGHT: {relevance_weight}
TOP N: {top_n}

Optimize for both relevance and diversity.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "optimized_recommendations": [
        {{
            "item_id": "item_001",
            "item_name": "Product F",
            "relevance_score": 0.92,
            "diversity_contribution": 0.78,
            "combined_score": 0.85,
            "category": "electronics",
            "reason": "High relevance with unique features"
        }},
        {{
            "item_id": "item_002",
            "item_name": "Product G",
            "relevance_score": 0.85,
            "diversity_contribution": 0.88,
            "combined_score": 0.865,
            "category": "books",
            "reason": "Different category for variety"
        }}
    ],
    "diversity_metrics": {{
        "intra_list_diversity": 0.82,
        "category_coverage": 0.75,
        "feature_spread": 0.80
    }},
    "optimization_method": "mmr|dpp|greedy",
    "relevance_preserved": 0.88,
    "diversity_gained": 0.35,
    "serendipity_score": 0.42
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
                "diversity_optimizer": result,
                "optimized_recommendations": result.get("optimized_recommendations", []),
                "diversity_metrics": result.get("diversity_metrics", {}),
                "optimization_method": result.get("optimization_method", "unknown")
            }
        except Exception as e:
            return {
                "diversity_optimizer": {"error": f"Failed diversity optimization: {str(e)}"},
                "optimized_recommendations": [],
                "optimization_method": "unknown"
            }