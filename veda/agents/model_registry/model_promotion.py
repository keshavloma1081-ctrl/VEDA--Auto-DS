"""
Agent 104: Model Promotion Agent
Manages model promotion through stages
"""
from typing import Dict, Any
import json
from .base_agent import ModelRegistryBaseAgent

class ModelPromotionAgent(ModelRegistryBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Manage model promotion workflow"""
        
        model_id = state.get('model_id', '')
        from_stage = state.get('from_stage', 'staging')
        to_stage = state.get('to_stage', 'production')
        
        prompt = f"""You are a model promotion expert.

MODEL ID: {model_id}
FROM STAGE: {from_stage}
TO STAGE: {to_stage}

Manage model promotion with validation checks.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "model_id": "{model_id}",
    "promotion_request": {{
        "from_stage": "{from_stage}",
        "to_stage": "{to_stage}",
        "requested_by": "ml_engineer",
        "requested_at": "2025-05-12T10:00:00Z"
    }},
    "validation_checks": [
        {{
            "check_name": "performance_threshold",
            "status": "passed",
            "threshold": 0.85,
            "actual_value": 0.92,
            "details": "Accuracy exceeds minimum threshold"
        }},
        {{
            "check_name": "data_drift",
            "status": "passed",
            "threshold": 0.3,
            "actual_value": 0.12,
            "details": "Minimal drift detected"
        }},
        {{
            "check_name": "integration_tests",
            "status": "passed",
            "tests_run": 25,
            "tests_passed": 25
        }}
    ],
    "approval_required": true,
    "approvers": ["senior_ml_engineer", "ml_lead"],
    "approved_by": ["senior_ml_engineer"],
    "promotion_status": "pending_approval|approved|rejected",
    "rollback_plan": {{
        "previous_model_id": "model_122_v2",
        "rollback_strategy": "blue_green",
        "estimated_rollback_time_minutes": 5
    }},
    "deployment_config": {{
        "traffic_split": {{"new_model": 0.1, "old_model": 0.9}},
        "canary_duration_hours": 2,
        "monitoring_alerts": ["latency", "error_rate", "prediction_drift"]
    }}
}}
"""
        
        try:
            response_text = self.call_llm(prompt, max_tokens=2500).strip()
            
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            return {
                "model_promotion": result,
                "promotion_status": result.get("promotion_status", "unknown"),
                "validation_checks": result.get("validation_checks", []),
                "approval_required": result.get("approval_required", False)
            }
        except Exception as e:
            return {
                "model_promotion": {"error": f"Failed promotion: {str(e)}"},
                "promotion_status": "error",
                "validation_checks": []
            }