"""
Agent 120: Multi-Objective Optimization Agent
Solves problems with multiple conflicting objectives
"""
from typing import Dict, Any
import json
from .base_agent import OptimizationBaseAgent

class MultiObjectiveAgent(OptimizationBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Solve multi-objective optimization"""
        
        objectives = state.get('objectives', ['minimize_cost', 'maximize_quality'])
        num_variables = state.get('num_variables', 15)
        algorithm = state.get('algorithm', 'NSGA-II')
        
        prompt = f"""You are a multi-objective optimization expert.

OBJECTIVES: {objectives}
NUMBER OF VARIABLES: {num_variables}
ALGORITHM: {algorithm}

Solve multi-objective optimization problem.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "problem_definition": {{
        "objectives": {objectives},
        "num_objectives": {len(objectives)},
        "num_variables": {num_variables},
        "objective_types": ["minimize", "maximize"]
    }},
    "algorithm_config": {{
        "algorithm": "{algorithm}",
        "population_size": 100,
        "generations": 200,
        "crossover_prob": 0.9,
        "mutation_prob": 0.1
    }},
    "pareto_front": {{
        "num_solutions": 50,
        "solutions": [
            {{"variables": [1.2, 3.5, 2.1], "objectives": [100, 0.95]}},
            {{"variables": [2.0, 2.8, 3.0], "objectives": [120, 0.92]}},
            {{"variables": [1.5, 4.0, 1.8], "objectives": [95, 0.97]}}
        ],
        "dominated_solutions": 450,
        "pareto_optimal": 50
    }},
    "convergence_metrics": {{
        "hypervolume": 0.85,
        "spacing": 0.12,
        "spread": 0.78,
        "convergence_generation": 150
    }},
    "trade_off_analysis": {{
        "conflict_level": "high|medium|low",
        "objective_correlations": {{"cost_quality": -0.85}},
        "preferred_region": "balanced",
        "knee_points": [
            {{"solution_id": 15, "cost": 105, "quality": 0.94}}
        ]
    }},
    "performance": {{
        "execution_time_seconds": 45.5,
        "evaluations": 20000,
        "convergence_speed": "fast"
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
                "multi_objective": result,
                "pareto_solutions": result.get("pareto_front", {}).get("num_solutions", 0),
                "hypervolume": result.get("convergence_metrics", {}).get("hypervolume", 0),
                "execution_time": result.get("performance", {}).get("execution_time_seconds", 0)
            }
        except Exception as e:
            return {
                "multi_objective": {"error": f"Failed multi-objective: {str(e)}"},
                "pareto_solutions": 0,
                "hypervolume": 0
            }