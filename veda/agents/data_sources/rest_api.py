"""
Agent 13: REST API Agent
Fetches data from web APIs with proper auth and rate limiting
"""
from typing import Dict, Any
import json
from .base_agent import BaseAgent

class RESTAPIAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently fetch data from REST APIs"""
        
        spec_path = state.get('spec_path', 'test_data/data_sources/apis/api_spec.json')
        try:
            with open(spec_path, 'r', encoding='utf-8') as f:
                spec_content = f.read()
        except:
            spec_content = "{}"
        
        prompt = f"""You are a REST API integration specialist.

API SPECIFICATION:
{spec_content}

DATA NEEDED: {state.get('data_requirements', 'List all active users')}

Your tasks:
1. Design the API request (method, headers, params)
2. Handle pagination if needed
3. Plan rate limit strategy

Return ONLY valid JSON (no markdown, no backticks):
{{
    "request_method": "GET",
    "headers": {{"Authorization": "Bearer ...", "Content-Type": "application/json"}},
    "query_params": {{"page": 1, "limit": 100}},
    "body": null,
    "pagination_strategy": "offset",
    "rate_limit_per_hour": 100,
    "expected_response_fields": ["id", "name", "email"],
    "error_handling": "retry with exponential backoff"
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
                "rest_api": result,
                "request_method": result.get("request_method", "GET"),
                "needs_pagination": "pagination_strategy" in result,
                "rate_limited": result.get("rate_limit_per_hour", 0) > 0
            }
        except Exception as e:
            return {
                "rest_api": {"error": f"Failed to plan API request: {str(e)}"},
                "request_method": "unknown"
            }