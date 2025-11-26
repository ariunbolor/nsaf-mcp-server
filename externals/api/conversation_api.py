"""
Conversation-Driven API
----------------------
API endpoints for the conversation-driven agent creation interface.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel
import json
import asyncio
from datetime import datetime

from ..core.conversation_driven_builder import ConversationDrivenBuilder
from ..utils.logging import get_logger
from .websocket_handler import websocket_manager
from .auth import get_current_user

logger = get_logger(__name__)
router = APIRouter()
conversation_builder = ConversationDrivenBuilder()

# ----- Pydantic Models -----

class StartConversationRequest(BaseModel):
    user_id: str

class StartConversationResponse(BaseModel):
    conversation_id: str
    status: str
    timestamp: str

class MessageRequest(BaseModel):
    conversation_id: str
    message: str

class MessageResponse(BaseModel):
    response: str
    next_stage: str
    options: Optional[List[str]] = None
    understanding: Optional[Dict[str, Any]] = None
    agent_plan: Optional[Dict[str, Any]] = None
    agent_config: Optional[Dict[str, Any]] = None
    agent: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SkillApprovalRequest(BaseModel):
    skill_id: str
    admin_config: Dict[str, Any]

class SkillRejectionRequest(BaseModel):
    skill_id: str
    reason: str

# ----- API Endpoints -----

@router.post("/conversations/start", response_model=StartConversationResponse)
async def start_conversation(request: StartConversationRequest, user = Depends(get_current_user)):
    """Start a new agent creation conversation"""
    try:
        conversation_id = await conversation_builder.start_conversation(request.user_id)
        
        return {
            "conversation_id": conversation_id,
            "status": "active",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting conversation: {str(e)}")

@router.post("/conversations/{conversation_id}/message", response_model=MessageResponse)
async def process_message(conversation_id: str, request: MessageRequest, user = Depends(get_current_user)):
    """Process a message in an agent creation conversation"""
    try:
        result = await conversation_builder.process_message(
            conversation_id=conversation_id,
            message=request.message
        )
        
        return result
    except ValueError as e:
        logger.error(f"Value error in conversation: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.get("/conversations/{conversation_id}", response_model=Dict[str, Any])
async def get_conversation(conversation_id: str, user = Depends(get_current_user)):
    """Get conversation details"""
    try:
        if conversation_id not in conversation_builder.conversations:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
            
        return conversation_builder.conversations[conversation_id]
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting conversation: {str(e)}")

@router.get("/conversations", response_model=List[Dict[str, Any]])
async def list_conversations(user = Depends(get_current_user)):
    """List all conversations for a user"""
    try:
        user_conversations = [
            {
                'id': conv_id,
                'user_id': conv['user_id'],
                'start_time': conv['start_time'],
                'status': conv['status'],
                'current_stage': conv['current_stage'],
                'agent_id': conv.get('agent_id')
            }
            for conv_id, conv in conversation_builder.conversations.items()
            if conv['user_id'] == user.id
        ]
        
        return user_conversations
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")

@router.get("/admin/pending-skills", response_model=Dict[str, Dict[str, Any]])
async def get_pending_skills(user = Depends(get_current_user)):
    """Get all skills pending admin approval"""
    try:
        if not user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
            
        return conversation_builder.get_pending_skills()
    except Exception as e:
        logger.error(f"Error getting pending skills: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting pending skills: {str(e)}")

@router.post("/admin/approve-skill/{skill_id}")
async def approve_skill(skill_id: str, request: SkillApprovalRequest, user = Depends(get_current_user)):
    """Approve a pending skill with admin-provided configuration"""
    try:
        if not user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
            
        success = conversation_builder.approve_skill(skill_id, request.admin_config)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Skill {skill_id} not found")
            
        return {"status": "success", "message": f"Skill {skill_id} approved"}
    except Exception as e:
        logger.error(f"Error approving skill: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error approving skill: {str(e)}")

@router.post("/admin/reject-skill/{skill_id}")
async def reject_skill(skill_id: str, request: SkillRejectionRequest, user = Depends(get_current_user)):
    """Reject a pending skill"""
    try:
        if not user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
            
        success = conversation_builder.reject_skill(skill_id, request.reason)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Skill {skill_id} not found")
            
        return {"status": "success", "message": f"Skill {skill_id} rejected"}
    except Exception as e:
        logger.error(f"Error rejecting skill: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error rejecting skill: {str(e)}")

# ----- WebSocket Endpoint -----

@router.websocket("/ws/conversation/{conversation_id}")
async def websocket_conversation(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for real-time conversation updates"""
    await websocket.accept()
    
    try:
        # Add connection to manager
        await websocket_manager.connect(websocket)
        
        # Check if conversation exists
        if conversation_id not in conversation_builder.conversations:
            await websocket.send_text(json.dumps({
                'type': 'error',
                'data': {
                    'message': f"Conversation {conversation_id} not found"
                }
            }))
            await websocket_manager.disconnect(websocket)
            return
            
        # Send initial conversation state
        conversation = conversation_builder.conversations[conversation_id]
        await websocket.send_text(json.dumps({
            'type': 'conversation_state',
            'data': {
                'conversation_id': conversation_id,
                'current_stage': conversation['current_stage'],
                'messages': conversation['messages']
            }
        }))
        
        # Listen for messages
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data['type'] == 'message':
                # Process message
                result = await conversation_builder.process_message(
                    conversation_id=conversation_id,
                    message=message_data['content']
                )
                
                # Send response
                await websocket.send_text(json.dumps({
                    'type': 'response',
                    'data': result
                }))
                
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_text(json.dumps({
            'type': 'error',
            'data': {
                'message': f"Error: {str(e)}"
            }
        }))
        await websocket_manager.disconnect(websocket)
