"""
Agent 98: Feature Versioning Agent
Manages feature versions and schema evolution
"""
from typing import Dict, Any
import json
from .base_agent import FeatureStoreBaseAgent

class FeatureVersioningAgent(FeatureStoreBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Manage feature versions"""
        
        feature_name = state.get('feature_name', '')
        action = state.get('action', 'create_version')  # create_version, get_version, compare
        version = state.get('version', 'v1')
        
        prompt = f"""You are a feature versioning expert.

FEATURE NAME: {feature_name}
ACTION: {action}
VERSION: {version}

Manage feature versioning and schema changes.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "feature_name": "{feature_name}",
    "current_version": "v3",
    "versions": [
        {{
            "version": "v1",
            "created_at": "2025-01-01T00:00:00Z",
            "schema": {{"type": "int64", "nullable": false}},
            "transformation": "raw",
            "deprecated": true,
            "models_using": []
        }},
        {{
            "version": "v2",
            "created_at": "2025-03-01T00:00:00Z",
            "schema": {{"type": "float64", "nullable": false}},
            "transformation": "normalized",
            "deprecated": false,
            "models_using": ["model_v1", "model_v2"]
        }},
        {{
            "version": "v3",
            "created_at": "2025-05-01T00:00:00Z",
            "schema": {{"type": "float64", "nullable": false}},
            "transformation": "standardized",
            "deprecated": false,
            "models_using": ["model_v3"]
        }}
    ],
    "schema_changes": [
        {{
            "from_version": "v1",
            "to_version": "v2",
            "change_type": "type_change",
            "description": "Changed from int to float for precision",
            "breaking_change": true
        }}
    ],
    "backward_compatible": false,
    "migration_required": true
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
                "feature_versioning": result,
                "current_version": result.get("current_version", ""),
                "versions": result.get("versions", []),
                "backward_compatible": result.get("backward_compatible", False)
            }
        except Exception as e:
            return {
                "feature_versioning": {"error": f"Failed versioning: {str(e)}"},
                "current_version": "",
                "versions": []
            }