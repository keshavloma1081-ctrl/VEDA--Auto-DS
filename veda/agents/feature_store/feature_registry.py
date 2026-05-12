"""
Agent 97: Feature Registry Agent
Manages feature catalog and metadata
"""
from typing import Dict, Any
import json
from .base_agent import FeatureStoreBaseAgent

class FeatureRegistryAgent(FeatureStoreBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Register and manage features"""
        
        action = state.get('action', 'register')  # register, search, update, delete
        feature_name = state.get('feature_name', '')
        feature_metadata = state.get('feature_metadata', {})
        
        prompt = f"""You are a feature store registry expert.

ACTION: {action}
FEATURE NAME: {feature_name}
FEATURE METADATA: {feature_metadata}

Manage feature registration and catalog.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "feature_id": "feature_user_age_123",
    "feature_name": "{feature_name}",
    "feature_type": "numerical|categorical|embedding|text",
    "data_type": "int64|float64|string|array",
    "description": "User age in years",
    "source_table": "users",
    "transformation": "raw|normalized|binned",
    "tags": ["user_features", "demographic"],
    "owner": "data_team",
    "created_at": "2025-05-12T10:00:00Z",
    "updated_at": "2025-05-12T10:00:00Z",
    "version": "v1",
    "status": "active|deprecated|testing",
    "dependencies": ["feature_user_signup_date"],
    "usage_count": 42,
    "quality_score": 0.95
}}
"""
        
        try:
            response_text = self.call_llm(prompt, max_tokens=1500).strip()
            
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            return {
                "feature_registry": result,
                "feature_id": result.get("feature_id", ""),
                "feature_name": result.get("feature_name", ""),
                "status": result.get("status", "unknown")
            }
        except Exception as e:
            return {
                "feature_registry": {"error": f"Failed registry operation: {str(e)}"},
                "feature_id": "",
                "status": "error"
            }