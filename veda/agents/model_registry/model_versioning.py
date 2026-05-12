"""
Agent 102: Model Versioning Agent
Manages model versions and metadata
"""
from typing import Dict, Any
import json
from .base_agent import ModelRegistryBaseAgent

class ModelVersioningAgent(ModelRegistryBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Manage model versions"""
        
        model_name = state.get('model_name', '')
        action = state.get('action', 'create_version')
        version = state.get('version', 'v1')
        
        prompt = f"""You are a model versioning expert.

MODEL NAME: {model_name}
ACTION: {action}
VERSION: {version}

Manage model versions with complete metadata.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "model_name": "{model_name}",
    "version": "{version}",
    "version_id": "model_123_v3",
    "created_at": "2025-05-12T10:00:00Z",
    "created_by": "data_team",
    "model_metadata": {{
        "algorithm": "xgboost",
        "framework": "sklearn",
        "framework_version": "1.3.0",
        "model_type": "classification"
    }},
    "training_metadata": {{
        "dataset_version": "v2",
        "training_duration_minutes": 45,
        "training_samples": 100000,
        "hyperparameters": {{"max_depth": 6, "learning_rate": 0.1}}
    }},
    "performance_metrics": {{
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.91,
        "f1_score": 0.90,
        "auc_roc": 0.94
    }},
    "artifacts": {{
        "model_file": "s3://models/model_v3.pkl",
        "config_file": "s3://models/config_v3.json",
        "preprocessing_pipeline": "s3://models/pipeline_v3.pkl"
    }},
    "status": "production",
    "tags": ["xgboost", "classification", "production_ready"],
    "parent_version": "v2",
    "changelog": "Improved feature engineering and hyperparameter tuning"
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
                "model_versioning": result,
                "version_id": result.get("version_id", ""),
                "status": result.get("status", "unknown"),
                "performance_metrics": result.get("performance_metrics", {})
            }
        except Exception as e:
            return {
                "model_versioning": {"error": f"Failed versioning: {str(e)}"},
                "version_id": "",
                "status": "error"
            }