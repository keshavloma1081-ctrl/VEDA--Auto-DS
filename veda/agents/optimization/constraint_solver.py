"""
Agent 119: Constraint Solver Agent
Solves constraint satisfaction problems
"""
from typing import Dict, Any
import json
from .base_agent import OptimizationBaseAgent

class ConstraintSolverAgent(OptimizationBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Solve constraint satisfaction problem"""
        
        problem_type = state.get('problem_type', 'scheduling')
        num_variables = state.get('num_variables', 20)
        num_constraints = state.get('num_constraints', 50)
        
        prompt = f"""You are a constraint satisfaction expert.

PROBLEM TYPE: {problem_type}
NUMBER OF VARIABLES: {num_variables}
NUMBER OF CONSTRAINTS: {num_constraints}

Solve constraint satisfaction problem.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "problem_definition": {{
        "problem_type": "{problem_type}",
        "num_variables": {num_variables},
        "num_constraints": {num_constraints},
        "constraint_types": ["equality", "inequality", "logical"]
    }},
    "variables": [
        {{"name": "x1", "domain": [1, 2, 3, 4, 5], "type": "integer"}},
        {{"name": "x2", "domain": [0.0, 10.0], "type": "continuous"}}
    ],
    "constraints": [
        {{"type": "all_different", "variables": ["x1", "x2", "x3"]}},
        {{"type": "linear", "expression": "x1 + x2 <= 10"}},
        {{"type": "logical", "expression": "if x1 > 3 then x2 < 5"}}
    ],
    "solution": {{
        "satisfiable": true,
        "solution": {{"x1": 3, "x2": 5.5, "x3": 7}},
        "num_solutions": 12,
        "all_solutions_found": false
    }},
    "solver_stats": {{
        "algorithm": "backtracking|forward_checking|arc_consistency",
        "search_nodes_explored": 450,
        "backtracks": 25,
        "solve_time_ms": 150,
        "optimality": "first_solution|all_solutions|optimal"
    }},
    "propagation_stats": {{
        "domain_reductions": 120,
        "constraint_propagations": 300,
        "conflicts_detected": 15
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
                "constraint_solver": result,
                "satisfiable": result.get("solution", {}).get("satisfiable", False),
                "num_solutions": result.get("solution", {}).get("num_solutions", 0),
                "solve_time_ms": result.get("solver_stats", {}).get("solve_time_ms", 0)
            }
        except Exception as e:
            return {
                "constraint_solver": {"error": f"Failed CSP: {str(e)}"},
                "satisfiable": False,
                "num_solutions": 0
            }