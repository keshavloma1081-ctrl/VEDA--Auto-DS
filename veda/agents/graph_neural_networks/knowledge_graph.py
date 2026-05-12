"""
Agent 116: Knowledge Graph Agent
Builds and reasons over knowledge graphs
"""
from typing import Dict, Any
import json
from .base_agent import GNNBaseAgent

class KnowledgeGraphAgent(GNNBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build and query knowledge graph"""
        
        domain = state.get('domain', 'biomedical')
        task = state.get('task', 'reasoning')
        num_entities = state.get('num_entities', 10000)
        
        prompt = f"""You are a knowledge graph expert.

DOMAIN: {domain}
TASK: {task}
NUMBER OF ENTITIES: {num_entities}

Build and reason over knowledge graph.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "knowledge_graph": {{
        "domain": "{domain}",
        "num_entities": {num_entities},
        "num_relations": 50,
        "num_triples": 50000,
        "ontology_depth": 5
    }},
    "entity_types": [
        {{"type": "Person", "count": 2000}},
        {{"type": "Organization", "count": 1500}},
        {{"type": "Location", "count": 3000}},
        {{"type": "Concept", "count": 3500}}
    ],
    "relation_types": [
        {{"relation": "works_at", "count": 5000, "symmetry": "asymmetric"}},
        {{"relation": "located_in", "count": 8000, "symmetry": "asymmetric"}},
        {{"relation": "related_to", "count": 12000, "symmetry": "symmetric"}}
    ],
    "reasoning_results": {{
        "inferred_triples": 2500,
        "confidence_avg": 0.85,
        "reasoning_method": "rule_based|embedding|neural",
        "inconsistencies_detected": 12
    }},
    "sample_triples": [
        {{
            "subject": "Albert_Einstein",
            "predicate": "works_at",
            "object": "Princeton_University",
            "confidence": 0.98
        }},
        {{
            "subject": "Princeton_University",
            "predicate": "located_in",
            "object": "New_Jersey",
            "confidence": 1.0
        }}
    ],
    "embedding_model": {{
        "algorithm": "TransE|RotatE|ComplEx",
        "embedding_dim": 200,
        "trained_epochs": 100,
        "link_prediction_accuracy": 0.89
    }},
    "query_capabilities": {{
        "single_hop": true,
        "multi_hop": true,
        "temporal_reasoning": false,
        "probabilistic_reasoning": true
    }},
    "quality_metrics": {{
        "completeness": 0.78,
        "consistency": 0.95,
        "coverage": 0.82
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
                "knowledge_graph": result,
                "num_entities": result.get("knowledge_graph", {}).get("num_entities", 0),
                "num_triples": result.get("knowledge_graph", {}).get("num_triples", 0),
                "inferred_triples": result.get("reasoning_results", {}).get("inferred_triples", 0)
            }
        except Exception as e:
            return {
                "knowledge_graph": {"error": f"Failed knowledge graph: {str(e)}"},
                "num_entities": 0,
                "num_triples": 0
            }