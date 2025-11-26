"""
Authentication Module
-------------------
Handles user authentication and authorization for the API.
"""

from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os
from pathlib import Path
import json

from ..utils.logging import get_logger

logger = get_logger(__name__)

# ----- Configuration -----

# Secret key for JWT token generation
# In production, this should be stored securely, not hardcoded
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ----- Models -----

class User(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: bool = False
    is_admin: bool = False

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# ----- User Database -----

# In a real application, this would be a database
# For simplicity, we're using a JSON file
USERS_FILE = Path(__file__).parent.parent / "data" / "users.json"

def get_users_db() -> Dict[str, Dict[str, Any]]:
    """Get users from the JSON file"""
    if not USERS_FILE.exists():
        # Create directory if it doesn't exist
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Create empty users file
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
        
        return {}
    
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users_db(users_db: Dict[str, Dict[str, Any]]):
    """Save users to the JSON file"""
    with open(USERS_FILE, "w") as f:
        json.dump(users_db, f, indent=2)

# ----- Authentication Functions -----

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def get_user(username: str) -> Optional[UserInDB]:
    """Get a user from the database"""
    users_db = get_users_db()
    
    if username in users_db:
        user_dict = users_db[username]
        return UserInDB(**user_dict)
    
    return None

def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user"""
    user = get_user(username)
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    return user

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get the current user from the token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            raise credentials_exception
        
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(token_data.username)
    
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get the current active user"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return current_user

# ----- User Management Functions -----

def create_user(username: str, email: str, password: str, full_name: Optional[str] = None, is_admin: bool = False) -> User:
    """Create a new user"""
    users_db = get_users_db()
    
    if username in users_db:
        raise ValueError(f"Username {username} already exists")
    
    # Check if email already exists
    for user in users_db.values():
        if user["email"] == email:
            raise ValueError(f"Email {email} already exists")
    
    # Create user
    user_id = f"user_{len(users_db) + 1}"
    hashed_password = get_password_hash(password)
    
    user_dict = {
        "id": user_id,
        "username": username,
        "email": email,
        "full_name": full_name,
        "disabled": False,
        "is_admin": is_admin,
        "hashed_password": hashed_password
    }
    
    users_db[username] = user_dict
    save_users_db(users_db)
    
    return User(**{k: v for k, v in user_dict.items() if k != "hashed_password"})

def update_user(username: str, **kwargs) -> User:
    """Update a user"""
    users_db = get_users_db()
    
    if username not in users_db:
        raise ValueError(f"Username {username} not found")
    
    user_dict = users_db[username]
    
    # Update fields
    for key, value in kwargs.items():
        if key == "password":
            user_dict["hashed_password"] = get_password_hash(value)
        elif key in user_dict:
            user_dict[key] = value
    
    users_db[username] = user_dict
    save_users_db(users_db)
    
    return User(**{k: v for k, v in user_dict.items() if k != "hashed_password"})

def delete_user(username: str) -> bool:
    """Delete a user"""
    users_db = get_users_db()
    
    if username not in users_db:
        return False
    
    del users_db[username]
    save_users_db(users_db)
    
    return True

def list_users() -> Dict[str, User]:
    """List all users"""
    users_db = get_users_db()
    
    return {
        username: User(**{k: v for k, v in user_dict.items() if k != "hashed_password"})
        for username, user_dict in users_db.items()
    }

# ----- Initialize Admin User -----

def init_admin_user():
    """Initialize the admin user if it doesn't exist"""
    try:
        admin_username = os.environ.get("ADMIN_USERNAME", "admin")
        admin_password = os.environ.get("ADMIN_PASSWORD", "admin")
        admin_email = os.environ.get("ADMIN_EMAIL", "admin@example.com")
        
        # Check if admin user exists
        users_db = get_users_db()
        
        if admin_username not in users_db:
            # Create admin user
            create_user(
                username=admin_username,
                email=admin_email,
                password=admin_password,
                full_name="Administrator",
                is_admin=True
            )
            
            logger.info(f"Admin user '{admin_username}' created")
    except Exception as e:
        logger.error(f"Error initializing admin user: {str(e)}")

# Initialize admin user when module is imported
init_admin_user()
