"""
Agent 124: Sample Size Calculator Agent
Calculates required sample sizes for experiments
"""
from typing import Dict, Any
import json
from .base_agent import ABTestingBaseAgent

class SampleSizeCalculatorAgent(ABTestingBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate required sample size"""
        
        baseline_rate = state.get('baseline_rate', 0.10)
        mde = state.get('mde', 0.15)  # minimum detectable effect
        power = state.get('power', 0.8)
        
        prompt = f"""You are a sample size calculation expert.

BASELINE RATE: {baseline_rate}
MINIMUM DETECTABLE EFFECT: {mde}
STATISTICAL POWER: {power}

Calculate required sample size.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "input_parameters": {{
        "baseline_conversion": {baseline_rate},
        "minimum_detectable_effect": {mde},
        "expected_improvement": {baseline_rate * (1 + mde)},
        "statistical_power": {power},
        "significance_level": 0.05
    }},
    "sample_size_calculation": {{
        "per_variant": 5000,
        "total_required": 10000,
        "method": "two_proportion_z_test",
        "continuity_correction": true
    }},
    "duration_estimates": {{
        "daily_traffic_1000": 10,
        "daily_traffic_5000": 2,
        "daily_traffic_10000": 1,
        "recommended_minimum_days": 7
    }},
    "sensitivity_analysis": {{
        "mde_10pct": {{"sample_size": 8000}},
        "mde_15pct": {{"sample_size": 5000}},
        "mde_20pct": {{"sample_size": 3500}}
    }},
    "power_analysis": {{
        "power_at_n3000": 0.65,
        "power_at_n5000": 0.8,
        "power_at_n8000": 0.95
    }},
    "recommendations": {{
        "recommended_sample_size": 5000,
        "safety_margin": 1.2,
        "final_recommendation": 6000,
        "rationale": "Accounts for sample ratio mismatch and dropouts"
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
                "sample_size_calculator": result,
                "per_variant": result.get("sample_size_calculation", {}).get("per_variant", 0),
                "total_required": result.get("sample_size_calculation", {}).get("total_required", 0),
                "recommended": result.get("recommendations", {}).get("final_recommendation", 0)
            }
        except Exception as e:
            return {
                "sample_size_calculator": {"error": f"Failed calculation: {str(e)}"},
                "per_variant": 0,
                "total_required": 0
            }