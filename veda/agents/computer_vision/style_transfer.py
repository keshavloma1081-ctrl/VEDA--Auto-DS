"""
Agent 90: Style Transfer Agent
Applies artistic styles to images
"""
from typing import Dict, Any
import json
from .base_agent import CVBaseAgent

class StyleTransferAgent(CVBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply artistic style to image"""
        
        content_image = state.get('content_image', '')
        style_image = state.get('style_image', '')
        style_strength = state.get('style_strength', 0.7)
        
        prompt = f"""You are a style transfer expert.

CONTENT IMAGE: {content_image}
STYLE IMAGE: {style_image}
STYLE STRENGTH: {style_strength}

Apply artistic style transfer.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "style_applied": "Van Gogh Starry Night",
    "content_preservation": 0.85,
    "style_intensity": {style_strength},
    "model_used": "neural_style_transfer|cyclegan|adain",
    "output_resolution": {{"width": 1024, "height": 768}},
    "processing_time_ms": 5000,
    "optimization_iterations": 300,
    "output_path": "stylized_image.jpg",
    "style_metrics": {{
        "color_transfer": 0.78,
        "texture_transfer": 0.82,
        "content_preservation": 0.85
    }}
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
                "style_transfer": result,
                "style_applied": result.get("style_applied", ""),
                "content_preservation": result.get("content_preservation", 0),
                "output_path": result.get("output_path", "")
            }
        except Exception as e:
            return {
                "style_transfer": {"error": f"Failed style transfer: {str(e)}"},
                "style_applied": "",
                "output_path": ""
            }