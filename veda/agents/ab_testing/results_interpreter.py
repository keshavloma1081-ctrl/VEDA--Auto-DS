"""
Agent 126: Results Interpreter Agent
Interprets and communicates experiment results
"""
from typing import Dict, Any
import json
from .base_agent import ABTestingBaseAgent

class ResultsInterpreterAgent(ABTestingBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret experiment results"""
        
        results = state.get('results', {})
        metric = state.get('metric', 'conversion_rate')
        
        prompt = f"""You are an experiment results interpretation expert.

RESULTS: {results}
PRIMARY METRIC: {metric}

Interpret experiment results and provide actionable recommendations.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "summary": {{
        "experiment_name": "Homepage_CTA_Test",
        "duration_days": 14,
        "total_users": 10000,
        "primary_metric": "{metric}",
        "winner": "treatment"
    }},
    "statistical_results": {{
        "control_rate": 0.12,
        "treatment_rate": 0.15,
        "absolute_lift": 0.03,
        "relative_lift": 0.25,
        "p_value": 0.004,
        "confidence_interval_95": [0.01, 0.05],
        "statistically_significant": true
    }},
    "business_impact": {{
        "estimated_annual_revenue_impact": 250000,
        "estimated_conversions_per_month": 500,
        "roi_percentage": 450,
        "payback_period_days": 30
    }},
    "segment_analysis": {{
        "mobile": {{"lift": 0.30, "significant": true}},
        "desktop": {{"lift": 0.20, "significant": true}},
        "new_users": {{"lift": 0.35, "significant": true}},
        "returning_users": {{"lift": 0.15, "significant": false}}
    }},
    "guardrail_metrics": {{
        "page_load_time": {{"change": 0.02, "acceptable": true}},
        "error_rate": {{"change": 0.001, "acceptable": true}},
        "bounce_rate": {{"change": -0.05, "acceptable": true}}
    }},
    "recommendations": {{
        "primary": "Deploy treatment to 100% of traffic",
        "secondary": [
            "Monitor closely for 1 week post-launch",
            "Consider A/A test to validate infrastructure",
            "Document learnings for future tests"
        ],
        "risk_level": "low",
        "confidence": "high"
    }},
    "key_insights": [
        "Treatment performs 25% better than control",
        "Effect is consistent across all major segments",
        "No negative impact on guardrail metrics",
        "Expected annual revenue increase of $250K"
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
                "results_interpreter": result,
                "winner": result.get("summary", {}).get("winner", ""),
                "relative_lift": result.get("statistical_results", {}).get("relative_lift", 0),
                "recommendation": result.get("recommendations", {}).get("primary", "")
            }
        except Exception as e:
            return {
                "results_interpreter": {"error": f"Failed interpretation: {str(e)}"},
                "winner": "",
                "relative_lift": 0
            }