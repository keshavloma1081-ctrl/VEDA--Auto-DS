"""
Agent 129: REST API Agent
Exposes VEDA workflows as REST endpoints
"""
from typing import Dict, Any
import json
from .base_agent import APIBaseAgent

class RestAPIAgent(APIBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate REST API endpoints for VEDA workflows"""
        
        workflow_type = state.get('workflow_type', 'ml_pipeline')
        endpoints = state.get('endpoints', ['predict', 'train', 'evaluate'])
        
        prompt = f"""You are a REST API design expert.

WORKFLOW TYPE: {workflow_type}
ENDPOINTS NEEDED: {endpoints}

Design production-grade REST API endpoints for VEDA ML system.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "api_specification": {{
        "base_url": "https://api.veda-ml.com/v1",
        "version": "1.0",
        "authentication": "Bearer Token (JWT)",
        "rate_limit": "1000 requests/hour"
    }},
    "endpoints": [
        {{
            "path": "/predict",
            "method": "POST",
            "description": "Make predictions on new data",
            "request_body": {{
                "model_id": "string",
                "data": "array",
                "return_probabilities": "boolean"
            }},
            "response": {{
                "predictions": "array",
                "probabilities": "array",
                "model_version": "string",
                "inference_time_ms": "number"
            }},
            "example_curl": "curl -X POST https://api.veda-ml.com/v1/predict -H 'Authorization: Bearer TOKEN' -d '{{\\"model_id\\": \\"model_123\\", \\"data\\": [[1,2,3]]}}'"
        }},
        {{
            "path": "/train",
            "method": "POST",
            "description": "Train a new model",
            "request_body": {{
                "dataset_url": "string",
                "goal": "string",
                "config": "object"
            }},
            "response": {{
                "job_id": "string",
                "status": "string",
                "estimated_time_minutes": "number"
            }}
        }},
        {{
            "path": "/models/{{model_id}}",
            "method": "GET",
            "description": "Get model metadata",
            "response": {{
                "model_id": "string",
                "version": "string",
                "created_at": "string",
                "metrics": "object"
            }}
        }},
        {{
            "path": "/health",
            "method": "GET",
            "description": "Health check endpoint",
            "response": {{
                "status": "healthy",
                "uptime_seconds": 86400,
                "active_agents": 128
            }}
        }}
    ],
    "error_codes": {{
        "400": "Bad Request - Invalid input",
        "401": "Unauthorized - Invalid token",
        "429": "Too Many Requests - Rate limit exceeded",
        "500": "Internal Server Error"
    }},
    "security": {{
        "authentication": "JWT Bearer Token",
        "rate_limiting": "1000 req/hour per token",
        "cors": "enabled",
        "https_only": true
    }},
    "monitoring": {{
        "metrics_endpoint": "/metrics",
        "health_check": "/health",
        "logs": "structured_json"
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
                "rest_api": result,
                "endpoints_count": len(result.get("endpoints", [])),
                "base_url": result.get("api_specification", {}).get("base_url", ""),
                "authentication": result.get("api_specification", {}).get("authentication", "")
            }
        except Exception as e:
            return {
                "rest_api": {"error": f"Failed API design: {str(e)}"},
                "endpoints_count": 0,
                "base_url": ""
            }