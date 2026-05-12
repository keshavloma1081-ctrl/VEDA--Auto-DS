"""
Agent 99: Feature Serving Agent
Serves features for online and offline inference
"""
from typing import Dict, Any
import json
from .base_agent import FeatureStoreBaseAgent

class FeatureServingAgent(FeatureStoreBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serve features for inference"""
        
        serving_mode = state.get('serving_mode', 'online')  # online, offline, batch
        entity_ids = state.get('entity_ids', [])
        feature_names = state.get('feature_names', [])
        
        prompt = f"""You are a feature serving expert.

SERVING MODE: {serving_mode}
ENTITY IDS: {entity_ids}
REQUESTED FEATURES: {feature_names}

Serve features with optimal performance.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "features": [
        {{
            "entity_id": "user_123",
            "features": {{
                "user_age": 35,
                "user_spend_30d": 450.50,
                "user_category_preference": "electronics"
            }},
            "feature_timestamp": "2025-05-12T10:00:00Z",
            "freshness_ms": 50
        }}
    ],
    "serving_mode": "{serving_mode}",
    "latency_ms": 15,
    "cache_hit_rate": 0.85,
    "data_sources": {{
        "redis": 0.75,
        "postgres": 0.20,
        "computed": 0.05
    }},
    "feature_freshness": {{
        "user_age": "real_time",
        "user_spend_30d": "5_minutes",
        "user_category_preference": "1_hour"
    }},
    "sla_met": true,
    "errors": []
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
                "feature_serving": result,
                "features": result.get("features", []),
                "latency_ms": result.get("latency_ms", 0),
                "cache_hit_rate": result.get("cache_hit_rate", 0)
            }
        except Exception as e:
            return {
                "feature_serving": {"error": f"Failed serving: {str(e)}"},
                "features": [],
                "latency_ms": 0
            }