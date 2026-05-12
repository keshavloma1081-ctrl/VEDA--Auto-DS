"""
Agent 123: Statistical Tester Agent
Performs statistical tests on experiment results
"""
from typing import Dict, Any
import json
from .base_agent import ABTestingBaseAgent

class StatisticalTesterAgent(ABTestingBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical tests"""
        
        test_type = state.get('test_type', 't_test')
        control_data = state.get('control_data', {})
        treatment_data = state.get('treatment_data', {})
        
        prompt = f"""You are a statistical testing expert.

TEST TYPE: {test_type}
CONTROL DATA: {control_data}
TREATMENT DATA: {treatment_data}

Perform rigorous statistical testing.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "test_configuration": {{
        "test_type": "{test_type}",
        "alpha": 0.05,
        "two_tailed": true,
        "assumptions_checked": ["normality", "equal_variance"]
    }},
    "descriptive_statistics": {{
        "control": {{
            "mean": 0.12,
            "std": 0.05,
            "n": 5000,
            "confidence_interval_95": [0.118, 0.122]
        }},
        "treatment": {{
            "mean": 0.15,
            "std": 0.06,
            "n": 5000,
            "confidence_interval_95": [0.147, 0.153]
        }}
    }},
    "test_results": {{
        "statistic": 2.85,
        "p_value": 0.004,
        "significant": true,
        "effect_size": 0.25,
        "confidence_interval": [0.01, 0.05]
    }},
    "interpretation": {{
        "conclusion": "Treatment performs significantly better than control",
        "practical_significance": true,
        "lift": 0.25,
        "lift_percentage": 25.0,
        "recommendation": "Deploy treatment to 100% traffic"
    }},
    "additional_tests": {{
        "mann_whitney": {{"p_value": 0.003, "significant": true}},
        "chi_square": {{"p_value": 0.005, "significant": true}},
        "bootstrap_ci": [0.015, 0.048]
    }},
    "power_analysis": {{
        "achieved_power": 0.92,
        "required_power": 0.8,
        "adequately_powered": true
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
                "statistical_tester": result,
                "p_value": result.get("test_results", {}).get("p_value", 1.0),
                "significant": result.get("test_results", {}).get("significant", False),
                "effect_size": result.get("test_results", {}).get("effect_size", 0)
            }
        except Exception as e:
            return {
                "statistical_tester": {"error": f"Failed statistical test: {str(e)}"},
                "p_value": 1.0,
                "significant": False
            }