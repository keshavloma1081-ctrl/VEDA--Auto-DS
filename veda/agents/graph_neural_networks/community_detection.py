"""
Agent 115: Community Detection Agent
Detects communities and subgroups in graphs
"""
from typing import Dict, Any
import json
from .base_agent import GNNBaseAgent

class CommunityDetectionAgent(GNNBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect communities in graph"""
        
        graph_type = state.get('graph_type', 'social_network')
        algorithm = state.get('algorithm', 'louvain')
        resolution = state.get('resolution', 1.0)
        
        prompt = f"""You are a community detection expert.

GRAPH TYPE: {graph_type}
ALGORITHM: {algorithm}
RESOLUTION: {resolution}

Detect communities in graph structure.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "algorithm": "{algorithm}",
    "graph_statistics": {{
        "num_nodes": 5000,
        "num_edges": 25000,
        "avg_degree": 5.0,
        "density": 0.001
    }},
    "community_structure": {{
        "num_communities": 12,
        "modularity": 0.82,
        "coverage": 0.95,
        "resolution": {resolution}
    }},
    "communities": [
        {{
            "community_id": 0,
            "size": 650,
            "internal_edges": 2500,
            "external_edges": 150,
            "modularity_contribution": 0.15,
            "density": 0.012,
            "key_nodes": ["node_45", "node_123", "node_789"]
        }},
        {{
            "community_id": 1,
            "size": 420,
            "internal_edges": 1800,
            "external_edges": 100,
            "modularity_contribution": 0.12,
            "density": 0.020,
            "key_nodes": ["node_234", "node_567"]
        }}
    ],
    "inter_community_structure": {{
        "bridge_nodes": ["node_100", "node_200"],
        "bridge_edges": 45,
        "community_connectivity": [[0, 15, 8], [15, 0, 12], [8, 12, 0]]
    }},
    "quality_metrics": {{
        "conductance": 0.08,
        "normalized_cut": 0.15,
        "community_cohesion": 0.88
    }},
    "hierarchical_structure": {{
        "levels": 3,
        "top_level_communities": 4,
        "nested": true
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
                "community_detection": result,
                "num_communities": result.get("community_structure", {}).get("num_communities", 0),
                "modularity": result.get("community_structure", {}).get("modularity", 0),
                "communities": result.get("communities", [])
            }
        except Exception as e:
            return {
                "community_detection": {"error": f"Failed community detection: {str(e)}"},
                "num_communities": 0,
                "modularity": 0
            }