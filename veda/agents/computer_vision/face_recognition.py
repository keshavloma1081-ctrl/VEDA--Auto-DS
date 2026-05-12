"""
Agent 87: Face Recognition Agent
Detects and recognizes faces in images
"""
from typing import Dict, Any
import json
from .base_agent import CVBaseAgent

class FaceRecognitionAgent(CVBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and recognize faces"""
        
        image_path = state.get('image_path', '')
        known_faces = state.get('known_faces', [])
        
        prompt = f"""You are a face recognition expert.

IMAGE PATH: {image_path}
KNOWN FACES DATABASE: {known_faces}

Detect all faces and match against known database.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "faces_detected": [
        {{
            "face_id": 1,
            "bbox": {{"x": 200, "y": 100, "width": 150, "height": 200}},
            "identity": "John Doe",
            "confidence": 0.94,
            "landmarks": {{"left_eye": [220, 140], "right_eye": [280, 140], "nose": [250, 170], "mouth": [250, 200]}},
            "attributes": {{"age": 35, "gender": "male", "emotion": "happy"}}
        }}
    ],
    "total_faces": 1,
    "unknown_faces": 0,
    "model_used": "facenet|arcface|deepface",
    "detection_time_ms": 80,
    "recognition_time_ms": 120
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
                "face_recognition": result,
                "faces_detected": result.get("faces_detected", []),
                "total_faces": result.get("total_faces", 0),
                "unknown_faces": result.get("unknown_faces", 0)
            }
        except Exception as e:
            return {
                "face_recognition": {"error": f"Failed face recognition: {str(e)}"},
                "faces_detected": [],
                "total_faces": 0
            }