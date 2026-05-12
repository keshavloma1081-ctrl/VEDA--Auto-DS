"""
Agent 12: SQL Database Agent
Connects to databases, generates queries, retrieves data
"""
from typing import Dict, Any
import json
from .base_agent import BaseAgent

class SQLDatabaseAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Query SQL databases intelligently"""
        
        schema_path = state.get('schema_path', 'test_data/data_sources/databases/ecommerce_schema.sql')
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_content = f.read()
        except:
            schema_content = "users(id, name, email), orders(id, user_id, amount, date)"
        
        prompt = f"""You are a SQL database expert.

DATABASE SCHEMA:
{schema_content}

USER QUERY: {state.get('user_query', 'Get top 10 customers by total order value')}

Your tasks:
1. Understand the natural language query
2. Generate optimized SQL query
3. Explain the query logic

Return ONLY valid JSON (no markdown, no backticks):
{{
    "sql_query": "SELECT ... FROM ...",
    "query_type": "SELECT",
    "expected_rows": "10-100",
    "execution_plan": "explanation",
    "performance_notes": "optimization suggestions",
    "suggested_indexes": ["index suggestions"],
    "estimated_time_ms": 50
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
                "sql_database": result,
                "query_type": result.get("query_type", "SELECT"),
                "needs_optimization": "performance_notes" in result,
                "safe_to_execute": result.get("query_type") == "SELECT"
            }
        except Exception as e:
            return {
                "sql_database": {"error": f"Failed to generate SQL: {str(e)}"},
                "query_type": "unknown"
            }