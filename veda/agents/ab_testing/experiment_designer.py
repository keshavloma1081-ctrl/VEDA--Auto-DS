"""
Agent 122: Experiment Designer Agent
Designs A/B tests and experiments
"""
from typing import Dict, Any
import json
from .base_agent import ABTestingBaseAgent

class ExperimentDesignerAgent(ABTestingBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Design A/B test experiment"""
        
        metric = state.get('metric', 'conversion_rate')
        variants = state.get('variants', 2)
        experiment_type = state.get('experiment_type', 'ab_test')
        
        prompt = f"""You are an experiment design expert.

PRIMARY METRIC: {metric}
NUMBER OF VARIANTS: {variants}
EXPERIMENT TYPE: {experiment_type}

Design a rigorous A/B test experiment.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "experiment_design": {{
        "experiment_name": "Homepage_CTA_Test",
        "experiment_type": "{experiment_type}",
        "hypothesis": "New CTA button will increase conversions by 15%",
        "variants": [
            {{"id": "control", "name": "Current Design", "traffic_allocation": 0.5}},
            {{"id": "treatment", "name": "New CTA Design", "traffic_allocation": 0.5}}
        ]
    }},
    "metrics": {{
        "primary_metric": "{metric}",
        "secondary_metrics": ["click_rate", "bounce_rate", "time_on_page"],
        "guardrail_metrics": ["page_load_time", "error_rate"]
    }},
    "sample_size": {{
        "required_per_variant": 5000,
        "total_required": 10000,
        "power": 0.8,
        "significance_level": 0.05,
        "minimum_detectable_effect": 0.15
    }},
    "duration": {{
        "estimated_days": 14,
        "daily_traffic": 1000,
        "seasonality_considered": true
    }},
    "randomization": {{
        "unit": "user",
        "method": "hash_based",
        "consistent_assignment": true,
        "stratification": ["device_type", "user_segment"]
    }},
    "validity_checks": {{
        "sample_ratio_mismatch": true,
        "novelty_effect": "monitored",
        "interaction_effects": false,
        "selection_bias": "controlled"
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
                "experiment_designer": result,
                "required_sample_size": result.get("sample_size", {}).get("total_required", 0),
                "estimated_duration": result.get("duration", {}).get("estimated_days", 0),
                "num_variants": len(result.get("experiment_design", {}).get("variants", []))
            }
        except Exception as e:
            return {
                "experiment_designer": {"error": f"Failed experiment design: {str(e)}"},
                "required_sample_size": 0,
                "estimated_duration": 0
            }