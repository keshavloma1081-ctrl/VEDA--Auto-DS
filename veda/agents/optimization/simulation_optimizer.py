"""
Agent 121: Simulation Optimizer Agent
Optimizes systems using simulation-based methods
"""
from typing import Dict, Any
import json
from .base_agent import OptimizationBaseAgent

class SimulationOptimizerAgent(OptimizationBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize using simulation"""
        
        system_type = state.get('system_type', 'manufacturing')
        simulation_runs = state.get('simulation_runs', 1000)
        optimization_method = state.get('optimization_method', 'response_surface')
        
        prompt = f"""You are a simulation optimization expert.

SYSTEM TYPE: {system_type}
SIMULATION RUNS: {simulation_runs}
OPTIMIZATION METHOD: {optimization_method}

Optimize system using simulation.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "system_definition": {{
        "system_type": "{system_type}",
        "state_variables": 12,
        "decision_variables": 8,
        "stochastic_elements": ["arrival_rate", "service_time", "failure_rate"]
    }},
    "simulation_config": {{
        "simulation_runs": {simulation_runs},
        "warmup_period": 100,
        "run_length": 1000,
        "replications": 30,
        "random_seed": 42
    }},
    "optimization_method": {{
        "algorithm": "{optimization_method}",
        "surrogate_model": "polynomial|kriging|neural_network",
        "sample_strategy": "latin_hypercube|random|sobol",
        "convergence_criterion": "relative_improvement"
    }},
    "optimization_results": {{
        "optimal_configuration": {{"var1": 5.5, "var2": 12.0, "var3": 8.5}},
        "optimal_performance": 0.92,
        "performance_std": 0.05,
        "confidence_interval_95": [0.89, 0.95]
    }},
    "simulation_statistics": {{
        "total_simulations": {simulation_runs},
        "successful_runs": 985,
        "failed_runs": 15,
        "avg_runtime_per_sim_seconds": 2.5
    }},
    "sensitivity_analysis": {{
        "most_sensitive_variable": "var2",
        "sensitivity_indices": {{"var1": 0.15, "var2": 0.65, "var3": 0.20}},
        "interaction_effects": "moderate"
    }},
    "performance_metrics": {{
        "total_optimization_time_minutes": 45,
        "improvement_over_baseline": 0.35,
        "robustness_score": 0.88
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
                "simulation_optimizer": result,
                "optimal_performance": result.get("optimization_results", {}).get("optimal_performance", 0),
                "total_simulations": result.get("simulation_statistics", {}).get("total_simulations", 0),
                "improvement": result.get("performance_metrics", {}).get("improvement_over_baseline", 0)
            }
        except Exception as e:
            return {
                "simulation_optimizer": {"error": f"Failed simulation optimization: {str(e)}"},
                "optimal_performance": 0,
                "total_simulations": 0
            }