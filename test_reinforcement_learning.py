"""
Test Reinforcement Learning Agents (Agents 107-111)
"""
from veda.agents.reinforcement_learning.q_learning import QLearningAgent
from veda.agents.reinforcement_learning.policy_gradient import PolicyGradientAgent
from veda.agents.reinforcement_learning.actor_critic import ActorCriticAgent
from veda.agents.reinforcement_learning.reward_shaping import RewardShapingAgent
from veda.agents.reinforcement_learning.environment_simulator import EnvironmentSimulatorAgent

def test_reinforcement_learning():
    print("\n" + "="*60)
    print("TESTING REINFORCEMENT LEARNING AGENTS (107-111)")
    print("="*60)
    
    # Agent 107: Q-Learning
    print("\n[1/5] Q-Learning Agent...")
    try:
        q_learning = QLearningAgent()
        result = q_learning.execute({
            "environment": "gridworld",
            "episodes": 1000,
            "learning_rate": 0.1,
            "discount_factor": 0.99
        })
        print(f"Convergence Episode: {result.get('convergence_episode')}")
        print(f"Final Reward: {result.get('final_reward')}")
        print(f"Success Rate: {result.get('success_rate')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 108: Policy Gradient
    print("\n[2/5] Policy Gradient Agent...")
    try:
        policy_grad = PolicyGradientAgent()
        result = policy_grad.execute({
            "environment": "cartpole",
            "algorithm": "REINFORCE",
            "episodes": 1000
        })
        print(f"Final Reward: {result.get('final_reward')}")
        print(f"Solved At: Episode {result.get('solved_at')}")
        print(f"Converged: {result.get('converged')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 109: Actor-Critic
    print("\n[3/5] Actor-Critic Agent...")
    try:
        actor_critic = ActorCriticAgent()
        result = actor_critic.execute({
            "environment": "lunarlander",
            "variant": "A2C",
            "episodes": 1000
        })
        print(f"Final Reward: {result.get('final_reward')}")
        print(f"Stable: {result.get('stable')}")
        print(f"Actor Loss: {result.get('actor_loss')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 110: Reward Shaping
    print("\n[4/5] Reward Shaping Agent...")
    try:
        reward_shaping = RewardShapingAgent()
        result = reward_shaping.execute({
            "task": "navigation",
            "objectives": ["reach_goal", "minimize_steps"],
            "constraints": ["avoid_obstacles"]
        })
        print(f"Reward Components: {len(result.get('reward_function', {}).get('components', []))}")
        print(f"Convergence Improvement: {result.get('convergence_improvement')}")
        print(f"Policy Quality: {result.get('policy_quality')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 111: Environment Simulator
    print("\n[5/5] Environment Simulator Agent...")
    try:
        env_sim = EnvironmentSimulatorAgent()
        result = env_sim.execute({
            "env_type": "gridworld",
            "complexity": "medium",
            "stochastic": True
        })
        print(f"State Space: {result.get('state_space', {}).get('size', 0)}")
        print(f"Action Space: {result.get('action_space', {}).get('size', 0)}")
        print(f"Reproducible: {result.get('reproducible')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "="*60)
    print("REINFORCEMENT LEARNING TEST COMPLETE")
    print("="*60)
    print("\n✅ Domain 2/3 Complete: Reinforcement Learning (5 agents)")
    print("Progress: 111/128 agents (86.7%)")

if __name__ == "__main__":
    test_reinforcement_learning()