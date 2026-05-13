"""
Agent 131: Retry Logic Agent
Implements exponential backoff for failed operations
"""
from typing import Dict, Any
import json
from .base_agent import ErrorHandlingBaseAgent

class RetryLogicAgent(ErrorHandlingBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Implement retry logic with exponential backoff"""
        
        operation = state.get('operation', 'api_call')
        max_retries = state.get('max_retries', 3)
        
        prompt = f"""You are a retry logic expert.

OPERATION: {operation}
MAX RETRIES: {max_retries}

Design retry logic with exponential backoff.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "retry_config": {{
        "operation": "{operation}",
        "max_retries": {max_retries},
        "initial_delay_ms": 100,
        "max_delay_ms": 10000,
        "backoff_multiplier": 2,
        "jitter": true
    }},
    "retry_strategy": {{
        "algorithm": "exponential_backoff_with_jitter",
        "retry_on_errors": ["timeout", "connection_error", "503", "429"],
        "do_not_retry_on": ["400", "401", "403", "404"],
        "idempotent_only": true
    }},
    "retry_schedule": [
        {{"attempt": 1, "delay_ms": 100, "jitter_ms": 50}},
        {{"attempt": 2, "delay_ms": 200, "jitter_ms": 100}},
        {{"attempt": 3, "delay_ms": 400, "jitter_ms": 200}}
    ],
    "execution_example": {{
        "attempt_1": {{"status": "failed", "error": "timeout", "delay_before_retry": 100}},
        "attempt_2": {{"status": "failed", "error": "timeout", "delay_before_retry": 200}},
        "attempt_3": {{"status": "success", "response_time_ms": 150}}
    }},
    "metrics": {{
        "total_operations": 1000,
        "first_attempt_success": 950,
        "retry_success": 45,
        "permanent_failures": 5,
        "success_rate": 0.995
    }},
    "best_practices": {{
        "use_jitter": "Prevents thundering herd",
        "respect_rate_limits": "Check 429 Retry-After header",
        "log_all_retries": "For debugging",
        "circuit_breaker_integration": "Stop retrying if circuit open"
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
                "retry_logic": result,
                "max_retries": result.get("retry_config", {}).get("max_retries", 0),
                "success_rate": result.get("metrics", {}).get("success_rate", 0),
                "retry_schedule": result.get("retry_schedule", [])
            }
        except Exception as e:
            return {
                "retry_logic": {"error": f"Failed retry logic: {str(e)}"},
                "max_retries": 0,
                "success_rate": 0
            }