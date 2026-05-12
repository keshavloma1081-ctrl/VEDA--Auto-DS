"""
Agent 84: Image Classification Agent
Classifies images into predefined categories
"""
from typing import Dict, Any
import json
from .base_agent import CVBaseAgent

class ImageClassificationAgent(CVBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Classify image into categories"""
        
        image_path = state.get('image_path', '')
        categories = state.get('categories', ['cat', 'dog', 'bird', 'car', 'person'])
        
        prompt = f"""You are an image classification expert.

IMAGE PATH: {image_path}
POSSIBLE CATEGORIES: {categories}

Analyze the image and classify it into one of the categories.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "predicted_class": "dog",
    "confidence": 0.95,
    "top_5_predictions": [
        {{"class": "dog", "confidence": 0.95}},
        {{"class": "cat", "confidence": 0.03}}
    ],
    "model_architecture": "resnet50|mobilenet|vit",
    "preprocessing_applied": ["resize", "normalize"],
    "inference_time_ms": 45
}}
"""
        
        try:
            response_text = self.call_llm(prompt, max_tokens=1000).strip()
            
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            return {
                "image_classification": result,
                "predicted_class": result.get("predicted_class", "unknown"),
                "confidence": result.get("confidence", 0),
                "top_predictions": result.get("top_5_predictions", [])
            }
        except Exception as e:
            return {
                "image_classification": {"error": f"Failed to classify image: {str(e)}"},
                "predicted_class": "unknown",
                "confidence": 0
            }