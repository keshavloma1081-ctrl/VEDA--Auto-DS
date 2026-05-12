"""
Agent 114: Graph Clustering Agent
Clusters nodes in graphs
"""
from typing import Dict, Any
import json
from .base_agent import GNNBaseAgent

class GraphClusteringAgent(GNNBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Cluster nodes in graph"""
        
        graph_type = state.get('graph_type', 'protein_network')
        num_clusters = state.get('num_clusters', 10)
        algorithm = state.get('algorithm', 'spectral')
        
        prompt = f"""You are a graph clustering expert.

GRAPH TYPE: {graph_type}
NUMBER OF CLUSTERS: {num_clusters}
ALGORITHM: {algorithm}

Perform graph clustering.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "clustering_algorithm": "{algorithm}",
    "graph_properties": {{
        "num_nodes": 2000,
        "num_edges": 10000,
        "modularity": 0.72
    }},
    "clustering_results": {{
        "num_clusters": {num_clusters},
        "cluster_sizes": [250, 180, 220, 190, 160, 200, 180, 170, 230, 220],
        "silhouette_score": 0.68,
        "modularity_score": 0.72,
        "conductance": 0.15
    }},
    "cluster_quality": {{
        "intra_cluster_density": 0.65,
        "inter_cluster_density": 0.08,
        "separation": 0.85,
        "cohesion": 0.78
    }},
    "clusters": [
        {{
            "cluster_id": 0,
            "size": 250,
            "centroid_node": "node_45",
            "avg_degree": 12.5,
            "description": "High connectivity cluster"
        }},
        {{
            "cluster_id": 1,
            "size": 180,
            "centroid_node": "node_123",
            "avg_degree": 8.3,
            "description": "Medium connectivity cluster"
        }}
    ],
    "features_used": ["node_degree", "betweenness_centrality", "eigenvector_centrality"]
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
                "graph_clustering": result,
                "num_clusters": result.get("clustering_results", {}).get("num_clusters", 0),
                "modularity_score": result.get("clustering_results", {}).get("modularity_score", 0),
                "silhouette_score": result.get("clustering_results", {}).get("silhouette_score", 0)
            }
        except Exception as e:
            return {
                "graph_clustering": {"error": f"Failed clustering: {str(e)}"},
                "num_clusters": 0,
                "modularity_score": 0
            }