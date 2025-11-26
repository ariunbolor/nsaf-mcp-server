"""
Authentication API
----------------
API endpoints for user authentication and management.
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from datetime import timedelta
import json

from .auth import (
    User, Token, authenticate_user, create_access_token,
    get_current_active_user, create_user, update_user,
    delete_user, list_users, ACCESS_TOKEN_EXPIRE_MINUTES
)
from ..utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# ----- Pydantic Models -----

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str = None

class UserUpdate(BaseModel):
    email: EmailStr = None
    password: str = None
    full_name: str = None
    disabled: bool = None

class UserResponse(BaseModel):
    id: str
    username: str
    email: EmailStr
    full_name: str = None
    disabled: bool = False
    is_admin: bool = False

# ----- API Endpoints -----

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login to get an access token"""
    user = authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user

@router.post("/users", response_model=UserResponse)
async def create_new_user(user_data: UserCreate, current_user: User = Depends(get_current_active_user)):
    """Create a new user (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to create users"
        )
    
    try:
        new_user = create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )
        
        return new_user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate):
    """Register a new user"""
    try:
        new_user = create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )
        
        return new_user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.put("/users/{username}", response_model=UserResponse)
async def update_user_info(username: str, user_data: UserUpdate, current_user: User = Depends(get_current_active_user)):
    """Update user information"""
    # Only allow users to update their own info, or admins to update any user
    if current_user.username != username and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user"
        )
    
    try:
        # Filter out None values
        update_data = {k: v for k, v in user_data.dict().items() if v is not None}
        
        updated_user = update_user(username, **update_data)
        
        return updated_user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.delete("/users/{username}")
async def delete_user_account(username: str, current_user: User = Depends(get_current_active_user)):
    """Delete a user account (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete users"
        )
    
    success = delete_user(username)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {username} not found"
        )
    
    return {"status": "success", "message": f"User {username} deleted"}

@router.get("/users", response_model=List[UserResponse])
async def get_all_users(current_user: User = Depends(get_current_active_user)):
    """Get all users (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to list users"
        )
    
    users = list_users()
    
    return list(users.values())

@router.post("/users/{username}/make-admin")
async def make_user_admin(username: str, current_user: User = Depends(get_current_active_user)):
    """Make a user an admin (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to make users admin"
        )
    
    try:
        updated_user = update_user(username, is_admin=True)
        
        return {"status": "success", "message": f"User {username} is now an admin"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
