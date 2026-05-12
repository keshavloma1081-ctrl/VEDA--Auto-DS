"""
Agent 15: Cloud Storage Agent
Accesses data from S3, GCS, Azure Blob Storage
"""
from typing import Dict, Any
import json
from .base_agent import BaseAgent

class CloudStorageAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Access cloud storage intelligently"""
        
        inventory_path = state.get('inventory_path', 'test_data/data_sources/cloud/s3_inventory.md')
        try:
            with open(inventory_path, 'r', encoding='utf-8') as f:
                inventory_content = f.read()
        except:
            inventory_content = "No inventory available"
        
        prompt = f"""You are a cloud storage access specialist.

CLOUD INVENTORY:
{inventory_content}

ACCESS TASK: {state.get('task', 'List all CSV files from last 7 days')}

Your tasks:
1. Plan the storage access strategy
2. Handle authentication properly
3. Filter files by pattern/date
4. Plan download/upload strategy

Return ONLY valid JSON (no markdown, no backticks):
{{
    "provider": "s3",
    "access_method": "boto3",
    "authentication": "IAM",
    "file_list": ["file1.csv", "file2.csv"],
    "total_size_mb": 10.5,
    "batch_strategy": "how to process",
    "estimated_time_seconds": 30,
    "cost_estimate_usd": 0.15
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
                "cloud_storage": result,
                "provider": result.get("provider", "unknown"),
                "files_found": len(result.get("file_list", [])),
                "total_size_mb": result.get("total_size_mb", 0)
            }
        except Exception as e:
            return {
                "cloud_storage": {"error": f"Failed to access cloud storage: {str(e)}"},
                "provider": "unknown"
            }