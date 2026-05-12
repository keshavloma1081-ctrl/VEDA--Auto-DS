"""
Test Feature Store Agents (Agents 97-101)
"""
from veda.agents.feature_store.feature_registry import FeatureRegistryAgent
from veda.agents.feature_store.feature_versioning import FeatureVersioningAgent
from veda.agents.feature_store.feature_serving import FeatureServingAgent
from veda.agents.feature_store.feature_monitoring import FeatureMonitoringAgent
from veda.agents.feature_store.feature_lineage import FeatureLineageAgent
import json

def test_feature_store():
    print("\n" + "="*60)
    print("TESTING FEATURE STORE AGENTS (97-101)")
    print("="*60)
    
    # Agent 97: Feature Registry
    print("\n[1/5] Feature Registry Agent...")
    try:
        registry = FeatureRegistryAgent()
        result = registry.execute({
            "action": "register",
            "feature_name": "user_age_normalized",
            "feature_metadata": {
                "type": "numerical",
                "description": "User age normalized to 0-1 range"
            }
        })
        print(f"Feature ID: {result.get('feature_id')}")
        print(f"Status: {result.get('status')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 98: Feature Versioning
    print("\n[2/5] Feature Versioning Agent...")
    try:
        versioning = FeatureVersioningAgent()
        result = versioning.execute({
            "feature_name": "user_age_normalized",
            "action": "get_version",
            "version": "v2"
        })
        print(f"Current Version: {result.get('current_version')}")
        print(f"Total Versions: {len(result.get('versions', []))}")
        print(f"Backward Compatible: {result.get('backward_compatible')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 99: Feature Serving
    print("\n[3/5] Feature Serving Agent...")
    try:
        serving = FeatureServingAgent()
        result = serving.execute({
            "serving_mode": "online",
            "entity_ids": ["user_123", "user_456"],
            "feature_names": ["user_age", "user_spend_30d"]
        })
        print(f"Features Served: {len(result.get('features', []))}")
        print(f"Latency: {result.get('latency_ms')}ms")
        print(f"Cache Hit Rate: {result.get('cache_hit_rate')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 100: Feature Monitoring
    print("\n[4/5] Feature Monitoring Agent...")
    try:
        monitoring = FeatureMonitoringAgent()
        result = monitoring.execute({
            "feature_name": "user_age_normalized",
            "time_window": "7d",
            "metrics": ["drift", "quality", "freshness"]
        })
        print(f"Health Status: {result.get('health_status')}")
        print(f"Drift Detected: {result.get('drift_detected')}")
        print(f"Quality Metrics: {result.get('quality_metrics')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 101: Feature Lineage
    print("\n[5/5] Feature Lineage Agent...")
    try:
        lineage = FeatureLineageAgent()
        result = lineage.execute({
            "feature_name": "user_age_normalized",
            "analysis_type": "full"
        })
        print(f"Upstream Dependencies: {len(result.get('upstream_dependencies', []))}")
        print(f"Downstream Consumers: {len(result.get('downstream_consumers', []))}")
        if result.get('impact_analysis'):
            print(f"Impact Analysis: {result['impact_analysis']}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "="*60)
    print("FEATURE STORE TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_feature_store()