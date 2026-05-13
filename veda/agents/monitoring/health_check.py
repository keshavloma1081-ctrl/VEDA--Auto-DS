"""
Agent 135: Health Check Agent
Monitors system health and uptime
"""
from typing import Dict, Any
import json
from .base_agent import MonitoringBaseAgent

class HealthCheckAgent(MonitoringBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        
        check_type = state.get('check_type', 'full')
        component = state.get('component', 'all')
        
        prompt = f"""You are a system health monitoring expert.

CHECK TYPE: {check_type}
COMPONENT: {component}

Perform comprehensive health check.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "health_status": {{
        "overall": "healthy",
        "timestamp": "2025-05-13T10:00:00Z",
        "uptime_seconds": 2592000,
        "uptime_percent": 99.95
    }},
    "component_health": [
        {{
            "component": "api_gateway",
            "status": "healthy",
            "response_time_ms": 15,
            "error_rate": 0.001,
            "last_check": "2025-05-13T10:00:00Z"
        }},
        {{
            "component": "groq_api",
            "status": "healthy",
            "response_time_ms": 450,
            "error_rate": 0.002,
            "rate_limit_utilization": 0.35
        }},
        {{
            "component": "master_planner",
            "status": "healthy",
            "active_workflows": 23,
            "queue_length": 5,
            "avg_processing_time_ms": 1200
        }},
        {{
            "component": "agent_pool",
            "status": "healthy",
            "total_agents": 135,
            "active_agents": 12,
            "idle_agents": 123
        }},
        {{
            "component": "mlflow",
            "status": "healthy",
            "storage_used_gb": 45,
            "storage_limit_gb": 500,
            "active_experiments": 8
        }},
        {{
            "component": "database",
            "status": "healthy",
            "connection_pool_utilization": 0.42,
            "query_avg_time_ms": 8
        }}
    ],
    "performance_metrics": {{
        "requests_per_second": 125,
        "avg_response_time_ms": 850,
        "p95_response_time_ms": 1800,
        "p99_response_time_ms": 3200,
        "error_rate": 0.005,
        "success_rate": 0.995
    }},
    "resource_utilization": {{
        "cpu_percent": 45,
        "memory_percent": 62,
        "disk_percent": 38,
        "network_mbps": 120
    }},
    "dependencies": [
        {{
            "service": "groq_api",
            "status": "healthy",
            "latency_ms": 450
        }},
        {{
            "service": "mlflow_tracking",
            "status": "healthy",
            "latency_ms": 25
        }},
        {{
            "service": "postgresql",
            "status": "healthy",
            "latency_ms": 8
        }}
    ],
    "recent_incidents": [],
    "alerts": {{
        "active_alerts": 0,
        "warnings": 1,
        "info": 2
    }},
    "recommendations": [
        {{
            "priority": "low",
            "recommendation": "Consider scaling up during peak hours (2-4 PM)",
            "impact": "Reduce p99 latency by 20%"
        }}
    ]
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
                "health_check": result,
                "overall_status": result.get("health_status", {}).get("overall", "unknown"),
                "uptime_percent": result.get("health_status", {}).get("uptime_percent", 0),
                "error_rate": result.get("performance_metrics", {}).get("error_rate", 0),
                "healthy_components": sum(1 for c in result.get("component_health", []) if c.get("status") == "healthy")
            }
        except Exception as e:
            return {
                "health_check": {"error": f"Failed health check: {str(e)}"},
                "overall_status": "error",
                "uptime_percent": 0
            }