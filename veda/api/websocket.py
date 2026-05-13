"""
WebSocket endpoint for real-time workflow updates
"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        # job_id -> set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        
        self.active_connections[job_id].add(websocket)
        print(f"✅ WebSocket connected for job {job_id}")
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        """Remove WebSocket connection"""
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            
            # Clean up empty sets
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        
        print(f"❌ WebSocket disconnected for job {job_id}")
    
    async def send_update(self, job_id: str, message: dict):
        """Send update to all clients watching a job"""
        if job_id not in self.active_connections:
            return
        
        # Send to all connected clients
        disconnected = set()
        
        for websocket in self.active_connections[job_id]:
            try:
                await websocket.send_json(message)
            except:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket, job_id)
    
    async def broadcast(self, message: dict):
        """Broadcast to all connections"""
        for job_id in list(self.active_connections.keys()):
            await self.send_update(job_id, message)


# Global connection manager
manager = ConnectionManager()


async def workflow_websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for workflow updates"""
    await manager.connect(websocket, job_id)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)


# Add to FastAPI app:
"""
from veda.api.websocket import workflow_websocket_endpoint

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await workflow_websocket_endpoint(websocket, job_id)
"""