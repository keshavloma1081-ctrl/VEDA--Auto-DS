"""
Test A/B Testing Agents (Agents 122-124)
"""
from veda.agents.ab_testing.experiment_designer import ExperimentDesignerAgent
from veda.agents.ab_testing.statistical_tester import StatisticalTesterAgent
from veda.agents.ab_testing.sample_size_calculator import SampleSizeCalculatorAgent

def test_ab_testing():
    print("\n" + "="*60)
    print("TESTING A/B TESTING AGENTS (122-124)")
    print("="*60)
    
    # Agent 122: Experiment Designer
    print("\n[1/3] Experiment Designer Agent...")
    try:
        designer = ExperimentDesignerAgent()
        result = designer.execute({
            "metric": "conversion_rate",
            "variants": 2,
            "experiment_type": "ab_test"
        })
        print(f"Required Sample Size: {result.get('required_sample_size')}")
        print(f"Estimated Duration: {result.get('estimated_duration')} days")
        print(f"Num Variants: {result.get('num_variants')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 123: Statistical Tester
    print("\n[2/3] Statistical Tester Agent...")
    try:
        tester = StatisticalTesterAgent()
        result = tester.execute({
            "test_type": "t_test",
            "control_data": {"mean": 0.12, "n": 5000},
            "treatment_data": {"mean": 0.15, "n": 5000}
        })
        print(f"P-Value: {result.get('p_value')}")
        print(f"Significant: {result.get('significant')}")
        print(f"Effect Size: {result.get('effect_size')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 124: Sample Size Calculator
    print("\n[3/3] Sample Size Calculator Agent...")
    try:
        calculator = SampleSizeCalculatorAgent()
        result = calculator.execute({
            "baseline_rate": 0.10,
            "mde": 0.15,
            "power": 0.8
        })
        print(f"Per Variant: {result.get('per_variant')}")
        print(f"Total Required: {result.get('total_required')}")
        print(f"Recommended: {result.get('recommended')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "="*60)
    print("A/B TESTING TEST COMPLETE (Partial - 3/5 agents)")
    print("="*60)
    print("Progress: 124/128 agents (96.9%)")
    print("\nNote: 2 more A/B Testing agents to be created")

if __name__ == "__main__":
    test_ab_testing()