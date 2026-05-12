"""
Agent 118: Genetic Algorithm Agent
Solves optimization using evolutionary algorithms
"""
from typing import Dict, Any
import json
from .base_agent import OptimizationBaseAgent

class GeneticAlgorithmAgent(OptimizationBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Solve using genetic algorithm"""
        
        problem_type = state.get('problem_type', 'combinatorial')
        population_size = state.get('population_size', 100)
        generations = state.get('generations', 500)
        
        prompt = f"""You are a genetic algorithm expert.

PROBLEM TYPE: {problem_type}
POPULATION SIZE: {population_size}
GENERATIONS: {generations}

Solve optimization using genetic algorithm.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "algorithm_config": {{
        "population_size": {population_size},
        "generations": {generations},
        "crossover_rate": 0.8,
        "mutation_rate": 0.01,
        "selection_method": "tournament|roulette|rank"
    }},
    "problem_definition": {{
        "problem_type": "{problem_type}",
        "chromosome_length": 50,
        "search_space_size": 1e15,
        "fitness_function": "custom_objective"
    }},
    "optimization_results": {{
        "best_fitness": 0.95,
        "best_solution": [1, 0, 1, 1, 0, 1, 0, 1],
        "convergence_generation": 350,
        "final_diversity": 0.45
    }},
    "evolution_statistics": {{
        "avg_fitness_per_generation": [0.3, 0.5, 0.7, 0.85, 0.92, 0.95],
        "best_fitness_per_generation": [0.4, 0.6, 0.75, 0.88, 0.93, 0.95],
        "diversity_trend": "decreasing"
    }},
    "genetic_operators": {{
        "crossover_type": "single_point|two_point|uniform",
        "mutation_type": "bit_flip|swap|inversion",
        "elitism": true,
        "elite_size": 5
    }},
    "performance_metrics": {{
        "total_evaluations": 50000,
        "execution_time_seconds": 12.5,
        "convergence_speed": "fast"
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
                "genetic_algorithm": result,
                "best_fitness": result.get("optimization_results", {}).get("best_fitness", 0),
                "convergence_generation": result.get("optimization_results", {}).get("convergence_generation", 0),
                "execution_time": result.get("performance_metrics", {}).get("execution_time_seconds", 0)
            }
        except Exception as e:
            return {
                "genetic_algorithm": {"error": f"Failed GA: {str(e)}"},
                "best_fitness": 0,
                "convergence_generation": 0
            }