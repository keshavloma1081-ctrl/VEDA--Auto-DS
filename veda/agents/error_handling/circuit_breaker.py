"""
Agent 130: Circuit Breaker Agent
Prevents cascading failures in distributed system
"""
from typing import Dict, Any
import json
from .base_agent import ErrorHandlingBaseAgent

class CircuitBreakerAgent(ErrorHandlingBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Implement circuit breaker pattern"""
        
        service_name = state.get('service_name', 'groq_api')
        failure_threshold = state.get('failure_threshold', 5)
        timeout_seconds = state.get('timeout_seconds', 60)
        
        prompt = f"""You are a circuit breaker pattern expert.

SERVICE: {service_name}
FAILURE THRESHOLD: {failure_threshold}
TIMEOUT: {timeout_seconds} seconds

Design circuit breaker to prevent cascading failures.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "circuit_breaker_config": {{
        "service_name": "{service_name}",
        "failure_threshold": {failure_threshold},
        "timeout_seconds": {timeout_seconds},
        "half_open_requests": 3
    }},
    "states": {{
        "closed": {{
            "description": "Normal operation, requests pass through",
            "behavior": "Track failures, open if threshold exceeded"
        }},
        "open": {{
            "description": "Circuit tripped, requests fail fast",
            "behavior": "Return cached/fallback response immediately"
        }},
        "half_open": {{
            "description": "Testing if service recovered",
            "behavior": "Allow limited requests to test recovery"
        }}
    }},
    "current_state": "closed",
    "metrics": {{
        "total_requests": 1000,
        "failed_requests": 2,
        "success_rate": 0.998,
        "consecutive_failures": 0,
        "last_failure_time": null,
        "circuit_open_count": 0
    }},
    "fallback_strategy": {{
        "type": "cached_response",
        "cache_duration_seconds": 300,
        "default_response": {{"status": "service_unavailable", "message": "Using cached data"}}
    }},
    "recovery_strategy": {{
        "type": "exponential_backoff",
        "initial_delay_seconds": 10,
        "max_delay_seconds": 300,
        "multiplier": 2
    }},
    "monitoring": {{
        "alert_on_open": true,
        "log_state_changes": true,
        "metrics_export": "prometheus"
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
                "circuit_breaker": result,
                "current_state": result.get("current_state", "unknown"),
                "success_rate": result.get("metrics", {}).get("success_rate", 0),
                "consecutive_failures": result.get("metrics", {}).get("consecutive_failures", 0)
            }
        except Exception as e:
            return {
                "circuit_breaker": {"error": f"Failed circuit breaker: {str(e)}"},
                "current_state": "error",
                "success_rate": 0
            }