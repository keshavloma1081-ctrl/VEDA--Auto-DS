"""
Agent 85: Object Detection Agent
Detects and localizes objects in images
"""
from typing import Dict, Any
import json
from .base_agent import CVBaseAgent

class ObjectDetectionAgent(CVBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect objects in image"""
        
        image_path = state.get('image_path', '')
        min_confidence = state.get('min_confidence', 0.5)
        
        prompt = f"""You are an object detection expert.

IMAGE PATH: {image_path}
MINIMUM CONFIDENCE: {min_confidence}

Detect all objects in the image with bounding boxes.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "detected_objects": [
        {{
            "class": "person",
            "confidence": 0.92,
            "bbox": {{"x": 100, "y": 50, "width": 200, "height": 300}},
            "object_id": 1
        }},
        {{
            "class": "car",
            "confidence": 0.87,
            "bbox": {{"x": 400, "y": 200, "width": 150, "height": 100}},
            "object_id": 2
        }}
    ],
    "total_objects": 2,
    "model_used": "yolov8|fasterrcnn|detectron2",
    "image_dimensions": {{"width": 1920, "height": 1080}},
    "inference_time_ms": 120
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
                "object_detection": result,
                "detected_objects": result.get("detected_objects", []),
                "total_objects": result.get("total_objects", 0),
                "model_used": result.get("model_used", "unknown")
            }
        except Exception as e:
            return {
                "object_detection": {"error": f"Failed to detect objects: {str(e)}"},
                "detected_objects": [],
                "total_objects": 0
            }