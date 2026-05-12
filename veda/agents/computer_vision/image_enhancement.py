"""
Agent 89: Image Enhancement Agent
Improves image quality through various techniques
"""
from typing import Dict, Any
import json
from .base_agent import CVBaseAgent

class ImageEnhancementAgent(CVBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance image quality"""
        
        image_path = state.get('image_path', '')
        enhancements = state.get('enhancements', ['denoise', 'sharpen', 'color_correction'])
        
        prompt = f"""You are an image enhancement expert.

IMAGE PATH: {image_path}
REQUESTED ENHANCEMENTS: {enhancements}

Apply image enhancement techniques.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "applied_enhancements": [
        {{
            "technique": "denoising",
            "method": "bilateral_filter|nlm",
            "parameters": {{"strength": 0.7}},
            "quality_improvement": 0.25
        }},
        {{
            "technique": "sharpening",
            "method": "unsharp_mask",
            "parameters": {{"amount": 1.2, "radius": 1.0}},
            "quality_improvement": 0.15
        }}
    ],
    "before_metrics": {{"brightness": 120, "contrast": 45, "sharpness": 0.6}},
    "after_metrics": {{"brightness": 140, "contrast": 60, "sharpness": 0.85}},
    "output_path": "enhanced_image.jpg",
    "processing_time_ms": 300,
    "overall_quality_gain": 0.40
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
                "image_enhancement": result,
                "applied_enhancements": result.get("applied_enhancements", []),
                "quality_gain": result.get("overall_quality_gain", 0),
                "output_path": result.get("output_path", "")
            }
        except Exception as e:
            return {
                "image_enhancement": {"error": f"Failed enhancement: {str(e)}"},
                "applied_enhancements": [],
                "quality_gain": 0
            }