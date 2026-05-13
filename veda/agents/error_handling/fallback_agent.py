"""
Agent 132: Fallback Agent
Provides graceful degradation when services fail
"""
from typing import Dict, Any
import json
from .base_agent import ErrorHandlingBaseAgent

class FallbackAgent(ErrorHandlingBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Implement fallback strategies"""
        
        service = state.get('service', 'prediction_service')
        fallback_type = state.get('fallback_type', 'cached')
        
        prompt = f"""You are a fallback strategy expert.

SERVICE: {service}
FALLBACK TYPE: {fallback_type}

Design graceful degradation strategies.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "fallback_strategies": [
        {{
            "priority": 1,
            "strategy": "cached_response",
            "description": "Return cached predictions from last successful call",
            "cache_ttl_seconds": 3600,
            "staleness_warning": true
        }},
        {{
            "priority": 2,
            "strategy": "default_model",
            "description": "Use simpler baseline model",
            "model": "logistic_regression_baseline",
            "accuracy_tradeoff": -0.05
        }},
        {{
            "priority": 3,
            "strategy": "rule_based",
            "description": "Use business rules instead of ML",
            "rules": ["if age > 60 then high_risk", "if income < 30k then approve_manually"]
        }},
        {{
            "priority": 4,
            "strategy": "graceful_failure",
            "description": "Return error with user-friendly message",
            "message": "Predictions temporarily unavailable, please try again"
        }}
    ],
    "current_fallback": null,
    "fallback_metrics": {{
        "primary_service_uptime": 0.998,
        "fallback_activations": 12,
        "cache_hit_rate": 0.85,
        "user_impact_score": 0.02
    }},
    "degradation_levels": {{
        "level_1": "Full functionality",
        "level_2": "Cached responses only",
        "level_3": "Baseline model only",
        "level_4": "Rule-based only",
        "level_5": "Service unavailable"
    }},
    "automatic_recovery": {{
        "health_check_interval_seconds": 30,
        "recovery_threshold": 3,
        "gradual_traffic_increase": true
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
                "fallback": result,
                "strategies_count": len(result.get("fallback_strategies", [])),
                "current_fallback": result.get("current_fallback", None),
                "uptime": result.get("fallback_metrics", {}).get("primary_service_uptime", 0)
            }
        except Exception as e:
            return {
                "fallback": {"error": f"Failed fallback: {str(e)}"},
                "strategies_count": 0,
                "current_fallback": None
            }