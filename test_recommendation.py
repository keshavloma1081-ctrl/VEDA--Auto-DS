"""
Test Recommendation Systems Agents (Agents 92-96)
"""
from veda.agents.recommendation.collaborative_filtering import CollaborativeFilteringAgent
from veda.agents.recommendation.content_based import ContentBasedAgent
from veda.agents.recommendation.hybrid_recommender import HybridRecommenderAgent
from veda.agents.recommendation.cold_start_handler import ColdStartHandlerAgent
from veda.agents.recommendation.diversity_optimizer import DiversityOptimizerAgent
import json

def test_recommendation_systems():
    print("\n" + "="*60)
    print("TESTING RECOMMENDATION SYSTEMS AGENTS (92-96)")
    print("="*60)
    
    # Agent 92: Collaborative Filtering
    print("\n[1/5] Collaborative Filtering Agent...")
    try:
        cf_agent = CollaborativeFilteringAgent()
        result = cf_agent.execute({
            "user_id": "user_123",
            "user_history": ["item_1", "item_5", "item_12"],
            "similarity_metric": "cosine",
            "top_n": 5
        })
        print(f"Recommendations: {len(result.get('recommendations', []))}")
        print(f"Algorithm: {result.get('algorithm')}")
        print(f"Diversity Score: {result.get('diversity_score')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 93: Content-Based
    print("\n[2/5] Content-Based Filtering Agent...")
    try:
        cb_agent = ContentBasedAgent()
        result = cb_agent.execute({
            "user_id": "user_123",
            "user_profile": {"genre": ["action", "sci-fi"], "rating_preference": "high"},
            "item_catalog": ["item_1", "item_2", "item_3"],
            "top_n": 5
        })
        print(f"Recommendations: {len(result.get('recommendations', []))}")
        print(f"Algorithm: {result.get('algorithm')}")
        print(f"Novelty Score: {result.get('novelty_score')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 94: Hybrid Recommender
    print("\n[3/5] Hybrid Recommender Agent...")
    try:
        hybrid_agent = HybridRecommenderAgent()
        result = hybrid_agent.execute({
            "user_id": "user_123",
            "cf_results": ["item_1", "item_2"],
            "cb_results": ["item_2", "item_3"],
            "weights": {"cf": 0.6, "cb": 0.4},
            "top_n": 5
        })
        print(f"Recommendations: {len(result.get('recommendations', []))}")
        print(f"Ensemble Method: {result.get('ensemble_method')}")
        print(f"Diversity Score: {result.get('diversity_score')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 95: Cold Start Handler
    print("\n[4/5] Cold Start Handler Agent...")
    try:
        cold_start = ColdStartHandlerAgent()
        result = cold_start.execute({
            "entity_type": "user",
            "entity_id": "new_user_456",
            "available_data": {"age": 25, "location": "US"},
            "strategy": "popularity"
        })
        print(f"Recommendations: {len(result.get('recommendations', []))}")
        print(f"Strategy: {result.get('strategy')}")
        print(f"Confidence: {result.get('confidence_level')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 96: Diversity Optimizer
    print("\n[5/5] Diversity Optimizer Agent...")
    try:
        diversity = DiversityOptimizerAgent()
        result = diversity.execute({
            "initial_recommendations": [
                {"item_id": "item_1", "score": 0.9},
                {"item_id": "item_2", "score": 0.85}
            ],
            "diversity_weight": 0.5,
            "relevance_weight": 0.5,
            "top_n": 5
        })
        print(f"Optimized Recommendations: {len(result.get('optimized_recommendations', []))}")
        print(f"Optimization Method: {result.get('optimization_method')}")
        if result.get('diversity_metrics'):
            print(f"Diversity Metrics: {result['diversity_metrics']}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION SYSTEMS TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_recommendation_systems()