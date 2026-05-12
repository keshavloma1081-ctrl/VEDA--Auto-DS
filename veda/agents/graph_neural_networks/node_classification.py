"""
Agent 112: Node Classification Agent
Classifies nodes in graph structures
"""
from typing import Dict, Any
import json
from .base_agent import GNNBaseAgent

class NodeClassificationAgent(GNNBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Classify nodes in graph"""
        
        graph_type = state.get('graph_type', 'social_network')
        num_nodes = state.get('num_nodes', 1000)
        num_classes = state.get('num_classes', 5)
        
        prompt = f"""You are a node classification expert.

GRAPH TYPE: {graph_type}
NUMBER OF NODES: {num_nodes}
NUMBER OF CLASSES: {num_classes}

Perform node classification on graph data.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "graph_properties": {{
        "num_nodes": {num_nodes},
        "num_edges": 5000,
        "avg_degree": 5.0,
        "graph_type": "{graph_type}"
    }},
    "model_architecture": {{
        "gnn_type": "GCN|GAT|GraphSAGE",
        "num_layers": 3,
        "hidden_dims": [128, 64, 32],
        "aggregation": "mean|sum|attention",
        "activation": "relu"
    }},
    "training_results": {{
        "train_accuracy": 0.92,
        "val_accuracy": 0.88,
        "test_accuracy": 0.87,
        "f1_score": 0.86,
        "epochs_trained": 200
    }},
    "classification_results": {{
        "num_classes": {num_classes},
        "class_distribution": {{"class_0": 200, "class_1": 180, "class_2": 220}},
        "confusion_matrix": [[180, 20], [15, 185]],
        "per_class_accuracy": {{"class_0": 0.90, "class_1": 0.92}}
    }},
    "graph_features_used": {{
        "node_features": ["degree", "centrality", "embedding"],
        "edge_features": ["weight", "type"],
        "neighborhood_aggregation": true
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
                "node_classification": result,
                "test_accuracy": result.get("training_results", {}).get("test_accuracy", 0),
                "f1_score": result.get("training_results", {}).get("f1_score", 0),
                "num_classes": result.get("classification_results", {}).get("num_classes", 0)
            }
        except Exception as e:
            return {
                "node_classification": {"error": f"Failed node classification: {str(e)}"},
                "test_accuracy": 0,
                "f1_score": 0
            }