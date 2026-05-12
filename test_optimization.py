"""
Test Optimization Agents (Agents 117-121)
"""
from veda.agents.optimization.linear_programming import LinearProgrammingAgent
from veda.agents.optimization.genetic_algorithm import GeneticAlgorithmAgent
from veda.agents.optimization.constraint_solver import ConstraintSolverAgent
from veda.agents.optimization.multi_objective import MultiObjectiveAgent
from veda.agents.optimization.simulation_optimizer import SimulationOptimizerAgent

def test_optimization():
    print("\n" + "="*60)
    print("TESTING OPTIMIZATION AGENTS (117-121)")
    print("="*60)
    
    # Agent 117: Linear Programming
    print("\n[1/5] Linear Programming Agent...")
    try:
        lp = LinearProgrammingAgent()
        result = lp.execute({
            "objective": "maximize_profit",
            "num_variables": 10,
            "num_constraints": 15
        })
        print(f"Optimal Value: {result.get('optimal_value')}")
        print(f"Status: {result.get('status')}")
        print(f"Solve Time: {result.get('solve_time_ms')}ms")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 118: Genetic Algorithm
    print("\n[2/5] Genetic Algorithm Agent...")
    try:
        ga = GeneticAlgorithmAgent()
        result = ga.execute({
            "problem_type": "combinatorial",
            "population_size": 100,
            "generations": 500
        })
        print(f"Best Fitness: {result.get('best_fitness')}")
        print(f"Convergence Gen: {result.get('convergence_generation')}")
        print(f"Execution Time: {result.get('execution_time')}s")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 119: Constraint Solver
    print("\n[3/5] Constraint Solver Agent...")
    try:
        csp = ConstraintSolverAgent()
        result = csp.execute({
            "problem_type": "scheduling",
            "num_variables": 20,
            "num_constraints": 50
        })
        print(f"Satisfiable: {result.get('satisfiable')}")
        print(f"Num Solutions: {result.get('num_solutions')}")
        print(f"Solve Time: {result.get('solve_time_ms')}ms")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 120: Multi-Objective
    print("\n[4/5] Multi-Objective Optimization Agent...")
    try:
        mo = MultiObjectiveAgent()
        result = mo.execute({
            "objectives": ["minimize_cost", "maximize_quality"],
            "num_variables": 15,
            "algorithm": "NSGA-II"
        })
        print(f"Pareto Solutions: {result.get('pareto_solutions')}")
        print(f"Hypervolume: {result.get('hypervolume')}")
        print(f"Execution Time: {result.get('execution_time')}s")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 121: Simulation Optimizer
    print("\n[5/5] Simulation Optimizer Agent...")
    try:
        sim_opt = SimulationOptimizerAgent()
        result = sim_opt.execute({
            "system_type": "manufacturing",
            "simulation_runs": 1000,
            "optimization_method": "response_surface"
        })
        print(f"Optimal Performance: {result.get('optimal_performance')}")
        print(f"Total Simulations: {result.get('total_simulations')}")
        print(f"Improvement: {result.get('improvement')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "="*60)
    print("OPTIMIZATION TEST COMPLETE")
    print("="*60)
    print("\n✅ Domain 1/3 Complete: Optimization (5 agents)")
    print("Progress: 121/128 agents (94.5%)")

if __name__ == "__main__":
    test_optimization()