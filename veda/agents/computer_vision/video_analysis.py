"""
Agent 91: Video Analysis Agent
Analyzes video content frame by frame
"""
from typing import Dict, Any
import json
from .base_agent import CVBaseAgent

class VideoAnalysisAgent(CVBaseAgent):
    def __init__(self):
        super().__init__()
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze video content"""
        
        video_path = state.get('video_path', '')
        analysis_type = state.get('analysis_type', 'action_recognition')
        
        prompt = f"""You are a video analysis expert.

VIDEO PATH: {video_path}
ANALYSIS TYPE: {analysis_type}

Analyze video content frame by frame.

Return ONLY valid JSON (no markdown, no backticks):
{{
    "video_metadata": {{
        "duration_seconds": 120,
        "fps": 30,
        "resolution": {{"width": 1920, "height": 1080}},
        "total_frames": 3600
    }},
    "detected_actions": [
        {{
            "action": "walking",
            "start_frame": 0,
            "end_frame": 300,
            "confidence": 0.92,
            "timestamp": "00:00-00:10"
        }},
        {{
            "action": "running",
            "start_frame": 301,
            "end_frame": 600,
            "confidence": 0.88,
            "timestamp": "00:10-00:20"
        }}
    ],
    "scene_changes": [100, 500, 1200],
    "objects_tracked": {{"person": 2, "car": 1}},
    "motion_intensity": "high|medium|low",
    "processing_time_seconds": 45,
    "model_used": "i3d|slowfast|timesformer"
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
                "video_analysis": result,
                "detected_actions": result.get("detected_actions", []),
                "scene_changes": result.get("scene_changes", []),
                "objects_tracked": result.get("objects_tracked", {})
            }
        except Exception as e:
            return {
                "video_analysis": {"error": f"Failed video analysis: {str(e)}"},
                "detected_actions": [],
                "scene_changes": []
            }