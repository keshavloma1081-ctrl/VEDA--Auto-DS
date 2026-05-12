"""
Agent 113: Link Prediction Agent
Predicts missing or future links in graphs
"""
from typing import Dict, Any
import json
from .base_agent import GNNBaseAgent

class LinkPredictionAgent(GNNBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict links in graph"""
        
        graph_type = state.get('graph_type', 'citation_network')
        num_nodes = state.get('num_nodes', 5000)
        prediction_task = state.get('prediction_task', 'missing_links')
        
        prompt = f"""You are a link prediction expert.

GRAPH TYPE: {graph_type}
NUMBER OF NODES: {num_nodes}
PREDICTION TASK: {prediction_task}

Predict links in graph structure.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "graph_statistics": {{
        "num_nodes": {num_nodes},
        "num_edges": 20000,
        "density": 0.0016,
        "avg_clustering_coefficient": 0.45
    }},
    "model_architecture": {{
        "encoder": "GCN|GAT|SAGE",
        "decoder": "inner_product|mlp",
        "embedding_dim": 128,
        "num_layers": 2
    }},
    "training_config": {{
        "negative_sampling_ratio": 1.0,
        "loss_function": "binary_cross_entropy",
        "optimizer": "adam",
        "learning_rate": 0.001
    }},
    "prediction_results": {{
        "auc_roc": 0.94,
        "precision": 0.88,
        "recall": 0.85,
        "f1_score": 0.86,
        "ap_score": 0.90
    }},
    "predicted_links": [
        {{
            "source": "node_123",
            "target": "node_456",
            "probability": 0.92,
            "confidence": "high"
        }},
        {{
            "source": "node_789",
            "target": "node_234",
            "probability": 0.85,
            "confidence": "medium"
        }}
    ],
    "feature_importance": {{
        "common_neighbors": 0.35,
        "node_embeddings": 0.45,
        "graph_structure": 0.20
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
                "link_prediction": result,
                "auc_roc": result.get("prediction_results", {}).get("auc_roc", 0),
                "predicted_links": result.get("predicted_links", []),
                "f1_score": result.get("prediction_results", {}).get("f1_score", 0)
            }
        except Exception as e:
            return {
                "link_prediction": {"error": f"Failed link prediction: {str(e)}"},
                "auc_roc": 0,
                "predicted_links": []
            }