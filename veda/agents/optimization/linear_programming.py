"""
Agent 117: Linear Programming Agent
Solves linear programming optimization problems
"""
from typing import Dict, Any
import json
from .base_agent import OptimizationBaseAgent

class LinearProgrammingAgent(OptimizationBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Solve linear programming problem"""
        
        objective = state.get('objective', 'maximize_profit')
        num_variables = state.get('num_variables', 10)
        num_constraints = state.get('num_constraints', 15)
        
        prompt = f"""You are a linear programming expert.

OBJECTIVE: {objective}
NUMBER OF VARIABLES: {num_variables}
NUMBER OF CONSTRAINTS: {num_constraints}

Solve linear programming optimization.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "problem_definition": {{
        "objective_type": "maximize|minimize",
        "objective_function": "3*x1 + 5*x2 + 2*x3",
        "num_variables": {num_variables},
        "num_constraints": {num_constraints}
    }},
    "constraints": [
        {{"type": "inequality", "expression": "2*x1 + 3*x2 <= 100"}},
        {{"type": "inequality", "expression": "x1 + x2 <= 50"}},
        {{"type": "equality", "expression": "x1 + 2*x2 = 30"}}
    ],
    "solution": {{
        "optimal_value": 245.5,
        "optimal_solution": {{"x1": 20, "x2": 15, "x3": 10}},
        "slack_variables": {{"s1": 5, "s2": 0}},
        "shadow_prices": {{"constraint_1": 1.5, "constraint_2": 0.5}}
    }},
    "solver_info": {{
        "solver": "simplex|interior_point",
        "iterations": 12,
        "solve_time_ms": 45,
        "status": "optimal|infeasible|unbounded"
    }},
    "sensitivity_analysis": {{
        "objective_coefficient_ranges": {{"x1": [2.5, 4.0], "x2": [4.0, 6.0]}},
        "rhs_ranges": {{"constraint_1": [90, 120], "constraint_2": [40, 60]}}
    }}
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
                "linear_programming": result,
                "optimal_value": result.get("solution", {}).get("optimal_value", 0),
                "status": result.get("solver_info", {}).get("status", "unknown"),
                "solve_time_ms": result.get("solver_info", {}).get("solve_time_ms", 0)
            }
        except Exception as e:
            return {
                "linear_programming": {"error": f"Failed LP: {str(e)}"},
                "optimal_value": 0,
                "status": "error"
            }