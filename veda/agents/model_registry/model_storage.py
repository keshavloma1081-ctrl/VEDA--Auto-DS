"""
Agent 103: Model Storage Agent
Manages model artifact storage and retrieval
"""
from typing import Dict, Any
import json
from .base_agent import ModelRegistryBaseAgent

class ModelStorageAgent(ModelRegistryBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Manage model artifact storage"""
        
        action = state.get('action', 'store')  # store, retrieve, delete
        model_id = state.get('model_id', '')
        storage_backend = state.get('storage_backend', 's3')
        
        prompt = f"""You are a model storage expert.

ACTION: {action}
MODEL ID: {model_id}
STORAGE BACKEND: {storage_backend}

Manage model artifact storage with optimal organization.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "model_id": "{model_id}",
    "storage_location": "s3://model-registry/models/{model_id}/",
    "artifacts": [
        {{
            "artifact_type": "model_binary",
            "path": "s3://model-registry/models/{model_id}/model.pkl",
            "size_mb": 45.2,
            "format": "pickle",
            "checksum": "sha256:abc123...",
            "compression": "gzip"
        }},
        {{
            "artifact_type": "preprocessing_pipeline",
            "path": "s3://model-registry/models/{model_id}/pipeline.pkl",
            "size_mb": 2.1,
            "format": "pickle",
            "checksum": "sha256:def456..."
        }},
        {{
            "artifact_type": "config",
            "path": "s3://model-registry/models/{model_id}/config.json",
            "size_mb": 0.01,
            "format": "json"
        }}
    ],
    "storage_backend": "{storage_backend}",
    "total_size_mb": 47.31,
    "redundancy": "replicated",
    "backup_locations": ["us-east-1", "us-west-2"],
    "access_control": {{
        "public": false,
        "allowed_roles": ["ml_engineer", "data_scientist"],
        "encryption": "AES-256"
    }},
    "retention_policy": "90_days",
    "storage_cost_monthly_usd": 0.15
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
                "model_storage": result,
                "storage_location": result.get("storage_location", ""),
                "total_size_mb": result.get("total_size_mb", 0),
                "artifacts": result.get("artifacts", [])
            }
        except Exception as e:
            return {
                "model_storage": {"error": f"Failed storage operation: {str(e)}"},
                "storage_location": "",
                "total_size_mb": 0
            }