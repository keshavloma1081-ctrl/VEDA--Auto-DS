"""
Agent 110: Reward Shaping Agent
Designs and optimizes reward functions for RL
"""
from typing import Dict, Any
import json
from .base_agent import RLBaseAgent

class RewardShapingAgent(RLBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Design optimal reward function"""
        
        task = state.get('task', 'navigation')
        objectives = state.get('objectives', ['reach_goal', 'minimize_steps'])
        constraints = state.get('constraints', ['avoid_obstacles'])
        
        prompt = f"""You are a reward shaping expert.

TASK: {task}
OBJECTIVES: {objectives}
CONSTRAINTS: {constraints}

Design an optimal reward function.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "task": "{task}",
    "reward_function": {{
        "type": "shaped_reward",
        "components": [
            {{
                "name": "goal_reward",
                "value": 100,
                "condition": "reached_goal",
                "weight": 1.0
            }},
            {{
                "name": "step_penalty",
                "value": -1,
                "condition": "each_step",
                "weight": 0.1
            }},
            {{
                "name": "obstacle_penalty",
                "value": -50,
                "condition": "collision",
                "weight": 1.0
            }},
            {{
                "name": "progress_reward",
                "value": 5,
                "condition": "distance_reduced",
                "weight": 0.3
            }}
        ],
        "total_function": "goal + progress - steps - obstacles"
    }},
    "shaping_strategy": {{
        "method": "potential_based|reward_engineering|inverse_rl",
        "ensures_optimal_policy": true,
        "avoids_reward_hacking": true
    }},
    "evaluation": {{
        "expected_convergence_speed": "30%_faster",
        "policy_quality": 0.95,
        "sparse_reward_problem": "solved",
        "exploration_efficiency": 0.88
    }},
    "hyperparameters": {{
        "reward_scaling": 0.01,
        "discount_factor_adjusted": 0.99,
        "normalization": "z_score"
    }},
    "potential_issues": {{
        "reward_hacking_risk": "low",
        "unintended_behaviors": [],
        "overfitting_to_reward": "low"
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
                "reward_shaping": result,
                "reward_function": result.get("reward_function", {}),
                "convergence_improvement": result.get("evaluation", {}).get("expected_convergence_speed", ""),
                "policy_quality": result.get("evaluation", {}).get("policy_quality", 0)
            }
        except Exception as e:
            return {
                "reward_shaping": {"error": f"Failed reward shaping: {str(e)}"},
                "reward_function": {},
                "policy_quality": 0
            }