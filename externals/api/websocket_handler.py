from fastapi import WebSocket
from typing import List, Dict, Any
import json
import asyncio
from datetime import datetime

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_times: Dict[WebSocket, datetime] = {}

    async def connect(self, websocket: WebSocket):
        """Add a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_times[websocket] = datetime.now()

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_times:
            del self.connection_times[websocket]

    async def broadcast(self, message: str):
        """Broadcast a message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)

        # Clean up any disconnected clients
        for connection in disconnected:
            await self.disconnect(connection)

    async def broadcast_json(self, data: Dict[str, Any]):
        """Broadcast a JSON message to all connected clients"""
        await self.broadcast(json.dumps(data))

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific client"""
        try:
            await websocket.send_text(message)
        except Exception:
            await self.disconnect(websocket)

    async def send_personal_json(self, data: Dict[str, Any], websocket: WebSocket):
        """Send a JSON message to a specific client"""
        await self.send_personal_message(json.dumps(data), websocket)

    async def broadcast_progress(self, stage: str, progress: int, details: str = None):
        """Broadcast a progress update"""
        await self.broadcast_json({
            'type': 'progress',
            'data': {
                'stage': stage,
                'progress': progress,
                'details': details
            }
        })

    async def broadcast_error(self, error: str, details: Dict[str, Any] = None):
        """Broadcast an error message"""
        await self.broadcast_json({
            'type': 'error',
            'data': {
                'message': error,
                'details': details
            }
        })

    async def broadcast_quantum_update(self, progress: int, details: str = None):
        """Broadcast a quantum computation update"""
        await self.broadcast_json({
            'type': 'quantum_update',
            'data': {
                'progress': progress,
                'details': details
            }
        })

    async def broadcast_reasoning_progress(self, thought: str, details: List[str] = None):
        """Broadcast a reasoning progress update"""
        await self.broadcast_json({
            'type': 'reasoning_progress',
            'data': {
                'thought': thought,
                'details': details
            }
        })

    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)

    def get_connection_time(self, websocket: WebSocket) -> datetime:
        """Get the connection time for a specific WebSocket"""
        return self.connection_times.get(websocket)

# Global WebSocket manager instance
websocket_manager = WebSocketManager()
