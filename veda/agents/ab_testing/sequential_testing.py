"""
Agent 125: Sequential Testing Agent
Performs sequential analysis and early stopping
"""
from typing import Dict, Any
import json
from .base_agent import ABTestingBaseAgent

class SequentialTestingAgent(ABTestingBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sequential testing"""
        
        current_sample = state.get('current_sample', 3000)
        method = state.get('method', 'sequential_probability_ratio')
        
        prompt = f"""You are a sequential testing expert.

CURRENT SAMPLE SIZE: {current_sample}
METHOD: {method}

Perform sequential analysis for early stopping.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "sequential_analysis": {{
        "method": "{method}",
        "current_sample_size": {current_sample},
        "max_sample_size": 10000,
        "alpha_spending_function": "obrien_fleming"
    }},
    "current_results": {{
        "control_conversion": 0.12,
        "treatment_conversion": 0.15,
        "observed_effect": 0.25,
        "z_score": 2.45
    }},
    "decision_boundaries": {{
        "upper_boundary": 2.8,
        "lower_boundary": -2.8,
        "futility_boundary": 0.5,
        "current_z": 2.45
    }},
    "stopping_decision": {{
        "stop_early": false,
        "reason": "not_significant_yet",
        "confidence": 0.92,
        "recommendation": "continue_to_n5000"
    }},
    "always_valid_p_value": {{
        "current_p_value": 0.014,
        "adjusted_alpha": 0.05,
        "valid_at_any_time": true
    }},
    "power_analysis": {{
        "current_power": 0.75,
        "target_power": 0.8,
        "probability_early_stop": 0.35
    }},
    "expected_sample_size": {{
        "under_null": 8000,
        "under_alternative": 6000,
        "savings_percentage": 40
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
                "sequential_testing": result,
                "stop_early": result.get("stopping_decision", {}).get("stop_early", False),
                "current_p_value": result.get("always_valid_p_value", {}).get("current_p_value", 1.0),
                "recommendation": result.get("stopping_decision", {}).get("recommendation", "")
            }
        except Exception as e:
            return {
                "sequential_testing": {"error": f"Failed sequential testing: {str(e)}"},
                "stop_early": False,
                "current_p_value": 1.0
            }