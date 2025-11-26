from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
import json
from typing import Dict, Any
from datetime import timedelta

from .agent_creation_handler import AgentCreationHandler
from .websocket_handler import websocket_manager
from .auth import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from .auth_api import router as auth_router
from .conversation_api import router as conversation_router
from .marketplace_api import router as marketplace_router

app = FastAPI(title="AI Agent Builder API", description="API for building and managing AI agents")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize handlers
agent_creation_handler = AgentCreationHandler()

# Include routers
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(conversation_router, prefix="/api", tags=["Conversation"])
app.include_router(marketplace_router, prefix="/api", tags=["Marketplace"])

# Token endpoint
@app.post("/api/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data['type'] == 'message':
                # Handle agent creation chat messages
                response = await agent_creation_handler.handle_chat_message(
                    message=message_data['content'],
                    current_step=message_data.get('currentStep', 'understanding_needs'),
                    agent_config=message_data.get('agentConfig', {})
                )
                
                await websocket.send_json(response)
                
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket)
    except Exception as e:
        error_response = {
            'success': False,
            'error': str(e)
        }
        await websocket.send_json(error_response)
        await websocket_manager.disconnect(websocket)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Legacy skill endpoints
@app.get("/api/skills", tags=["Skills"])
async def list_skills():
    """List all available skills"""
    skills = agent_creation_handler.skill_registry.list_skills()
    return {"skills": skills}

@app.get("/api/skills/{skill_id}", tags=["Skills"])
async def get_skill(skill_id: str):
    """Get details of a specific skill"""
    skill = agent_creation_handler.skill_registry.get_skill(skill_id)
    if skill:
        return skill
    return {"error": "Skill not found"}

@app.get("/api/pending-skills", tags=["Skills"])
async def list_pending_skills():
    """List all skills pending admin approval"""
    return {"pending_skills": agent_creation_handler.get_pending_skills()}

@app.post("/api/skills/{skill_id}/approve", tags=["Skills"])
async def approve_skill(skill_id: str, admin_config: Dict[str, Any]):
    """Approve a pending skill"""
    success = agent_creation_handler.approve_skill(skill_id, admin_config)
    if success:
        return {"message": "Skill approved successfully"}
    return {"error": "Failed to approve skill"}

@app.post("/api/skills/{skill_id}/reject", tags=["Skills"])
async def reject_skill(skill_id: str, reason: str):
    """Reject a pending skill"""
    success = agent_creation_handler.reject_skill(skill_id, reason)
    if success:
        return {"message": "Skill rejected successfully"}
    return {"error": "Failed to reject skill"}

@app.get("/api/version", tags=["System"])
async def get_version():
    """Get API version information"""
    return {
        "version": "1.0.0",
        "name": "AI Agent Builder API",
        "features": [
            "Conversation-driven agent creation",
            "Quantum symbolic reasoning",
            "Agent marketplace",
            "User authentication and authorization"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
