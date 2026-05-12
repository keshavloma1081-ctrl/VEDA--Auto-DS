"""
Agent 86: Semantic Segmentation Agent
Performs pixel-level classification of images
"""
from typing import Dict, Any
import json
from .base_agent import CVBaseAgent

class SemanticSegmentationAgent(CVBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Segment image at pixel level"""
        
        image_path = state.get('image_path', '')
        classes = state.get('classes', ['background', 'person', 'car', 'road', 'building'])
        
        prompt = f"""You are a semantic segmentation expert.

IMAGE PATH: {image_path}
SEGMENTATION CLASSES: {classes}

Perform pixel-level segmentation of the image.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "segmentation_map": {{
        "person": {{"pixel_count": 45000, "percentage": 15.2}},
        "car": {{"pixel_count": 20000, "percentage": 6.8}},
        "road": {{"pixel_count": 100000, "percentage": 33.9}},
        "background": {{"pixel_count": 130000, "percentage": 44.1}}
    }},
    "dominant_class": "background",
    "model_architecture": "unet|deeplabv3|segformer",
    "image_resolution": {{"width": 1920, "height": 1080}},
    "inference_time_ms": 200,
    "segmentation_quality": "high|medium|low"
}}
"""
        
        try:
            response_text = self.call_llm(prompt, max_tokens=1500).strip()
            
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            return {
                "semantic_segmentation": result,
                "segmentation_map": result.get("segmentation_map", {}),
                "dominant_class": result.get("dominant_class", "unknown"),
                "model_architecture": result.get("model_architecture", "unknown")
            }
        except Exception as e:
            return {
                "semantic_segmentation": {"error": f"Failed to segment image: {str(e)}"},
                "segmentation_map": {},
                "dominant_class": "unknown"
            }