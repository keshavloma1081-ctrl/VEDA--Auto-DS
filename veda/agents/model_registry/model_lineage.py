"""
Agent 106: Model Lineage Agent
Tracks model lineage, training data, and dependencies
"""
from typing import Dict, Any
import json
from .base_agent import ModelRegistryBaseAgent

class ModelLineageAgent(ModelRegistryBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Track complete model lineage"""
        
        model_id = state.get('model_id', '')
        lineage_depth = state.get('lineage_depth', 'full')
        
        prompt = f"""You are a model lineage expert.

MODEL ID: {model_id}
LINEAGE DEPTH: {lineage_depth}

Track complete model lineage and dependencies.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "model_id": "{model_id}",
    "model_name": "churn_prediction_model",
    "version": "v3",
    "training_lineage": {{
        "training_dataset": {{
            "dataset_id": "customer_data_v5",
            "dataset_version": "v5",
            "rows": 100000,
            "features": 45,
            "time_range": "2024-01-01 to 2025-04-01",
            "source_tables": ["customers", "transactions", "interactions"]
        }},
        "features_used": [
            {{
                "feature_id": "feature_user_age",
                "feature_version": "v2",
                "importance": 0.23
            }},
            {{
                "feature_id": "feature_spend_30d",
                "feature_version": "v3",
                "importance": 0.18
            }}
        ],
        "training_code": {{
            "git_repo": "github.com/company/ml-models",
            "commit_hash": "abc123def456",
            "branch": "main",
            "training_script": "train_churn_model.py"
        }},
        "training_environment": {{
            "framework": "sklearn==1.3.0",
            "python_version": "3.10",
            "compute": "AWS EC2 p3.2xlarge",
            "training_duration_minutes": 45
        }}
    }},
    "model_dependencies": [
        {{
            "dependency_type": "preprocessing_model",
            "model_id": "text_encoder_v2",
            "version": "v2"
        }},
        {{
            "dependency_type": "ensemble_component",
            "model_id": "base_classifier_v1",
            "version": "v1"
        }}
    ],
    "downstream_models": [
        {{
            "model_id": "ensemble_churn_model_v1",
            "model_name": "Ensemble Churn Predictor",
            "usage_type": "component"
        }}
    ],
    "experiment_tracking": {{
        "experiment_id": "exp_456",
        "mlflow_run_id": "run_789",
        "hyperparameter_search": {{
            "search_space": {{"max_depth": [3, 6, 9], "learning_rate": [0.01, 0.1, 0.3]}},
            "best_params": {{"max_depth": 6, "learning_rate": 0.1}},
            "trials": 27
        }}
    }},
    "lineage_graph": {{
        "nodes": [
            {{"id": "raw_data", "type": "data"}},
            {{"id": "feature_eng", "type": "process"}},
            {{"id": "model_v3", "type": "model"}},
            {{"id": "production", "type": "deployment"}}
        ],
        "edges": [
            {{"from": "raw_data", "to": "feature_eng"}},
            {{"from": "feature_eng", "to": "model_v3"}},
            {{"from": "model_v3", "to": "production"}}
        ]
    }},
    "reproducibility_score": 0.95
}}
"""
        
        try:
            response_text = self.call_llm(prompt, max_tokens=3000).strip()
            
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            return {
                "model_lineage": result,
                "training_lineage": result.get("training_lineage", {}),
                "model_dependencies": result.get("model_dependencies", []),
                "reproducibility_score": result.get("reproducibility_score", 0)
            }
        except Exception as e:
            return {
                "model_lineage": {"error": f"Failed lineage tracking: {str(e)}"},
                "training_lineage": {},
                "model_dependencies": []
            }