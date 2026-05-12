"""
Agent 101: Feature Lineage Agent
Tracks feature lineage, dependencies, and impact analysis
"""
from typing import Dict, Any
import json
from .base_agent import FeatureStoreBaseAgent

class FeatureLineageAgent(FeatureStoreBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Track feature lineage and dependencies"""
        
        feature_name = state.get('feature_name', '')
        analysis_type = state.get('analysis_type', 'full')  # upstream, downstream, full, impact
        
        prompt = f"""You are a feature lineage expert.

FEATURE NAME: {feature_name}
ANALYSIS TYPE: {analysis_type}

Track feature lineage, dependencies, and impact.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "feature_name": "{feature_name}",
    "upstream_dependencies": [
        {{
            "name": "raw_user_data",
            "type": "data_source",
            "table": "users",
            "columns": ["age", "signup_date"],
            "dependency_type": "direct"
        }},
        {{
            "name": "feature_user_tenure",
            "type": "feature",
            "version": "v2",
            "dependency_type": "derived"
        }}
    ],
    "downstream_consumers": [
        {{
            "name": "churn_prediction_model_v3",
            "type": "model",
            "version": "v3",
            "criticality": "high",
            "last_used": "2025-05-12T09:00:00Z"
        }},
        {{
            "name": "recommendation_model_v2",
            "type": "model",
            "version": "v2",
            "criticality": "medium",
            "last_used": "2025-05-11T15:00:00Z"
        }}
    ],
    "transformation_pipeline": [
        {{
            "step": 1,
            "operation": "extract",
            "source": "users.age",
            "transformation": "raw"
        }},
        {{
            "step": 2,
            "operation": "transform",
            "function": "normalize",
            "parameters": {{"method": "min_max"}}
        }},
        {{
            "step": 3,
            "operation": "validate",
            "rules": ["non_null", "range_18_100"]
        }}
    ],
    "impact_analysis": {{
        "models_affected": 2,
        "features_affected": 5,
        "risk_level": "medium",
        "estimated_retraining_cost": "2_hours"
    }},
    "lineage_graph": {{
        "nodes": ["raw_data", "feature_user_tenure", "feature_user_age", "model_v3"],
        "edges": [
            ["raw_data", "feature_user_tenure"],
            ["feature_user_tenure", "feature_user_age"],
            ["feature_user_age", "model_v3"]
        ]
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
                "feature_lineage": result,
                "upstream_dependencies": result.get("upstream_dependencies", []),
                "downstream_consumers": result.get("downstream_consumers", []),
                "impact_analysis": result.get("impact_analysis", {})
            }
        except Exception as e:
            return {
                "feature_lineage": {"error": f"Failed lineage tracking: {str(e)}"},
                "upstream_dependencies": [],
                "downstream_consumers": []
            }