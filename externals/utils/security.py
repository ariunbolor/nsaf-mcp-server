"""
Security Module
------------
Security utilities for the agent builder system.
"""

from typing import Dict, Optional, List, Set, Any, Union
import hashlib
import secrets
import time
import jwt
from datetime import datetime, timedelta
import re
from pathlib import Path
import json

from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    SecurityError
)
from .encryption import key_manager
from .logging import logger

class PasswordHasher:
    """Password hashing utility"""
    
    def __init__(self,
                 rounds: int = 100000,
                 salt_size: int = 32):
        self.rounds = rounds
        self.salt_size = salt_size
        
    def hash_password(self, password: str) -> Dict[str, str]:
        """
        Hash password using PBKDF2-SHA256
        
        Args:
            password: Password to hash
            
        Returns:
            Dictionary with hash and salt
        """
        try:
            salt = secrets.token_bytes(self.salt_size)
            hash_bytes = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                salt,
                self.rounds
            )
            
            return {
                'hash': hash_bytes.hex(),
                'salt': salt.hex(),
                'rounds': self.rounds
            }
        except Exception as e:
            raise SecurityError(f"Password hashing failed: {str(e)}")
            
    def verify_password(self,
                       password: str,
                       hash_dict: Dict[str, str]) -> bool:
        """
        Verify password against hash
        
        Args:
            password: Password to verify
            hash_dict: Dictionary with hash info
            
        Returns:
            True if password matches
        """
        try:
            salt = bytes.fromhex(hash_dict['salt'])
            stored_hash = bytes.fromhex(hash_dict['hash'])
            rounds = hash_dict.get('rounds', self.rounds)
            
            hash_bytes = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                salt,
                rounds
            )
            
            return secrets.compare_digest(hash_bytes, stored_hash)
            
        except Exception as e:
            raise SecurityError(f"Password verification failed: {str(e)}")

class TokenManager:
    """JWT token manager"""
    
    def __init__(self,
                 secret_key: Optional[str] = None,
                 token_expiry: int = 3600,  # 1 hour
                 refresh_expiry: int = 86400 * 7):  # 7 days
        self.secret_key = secret_key or secrets.token_hex(32)
        self.token_expiry = token_expiry
        self.refresh_expiry = refresh_expiry
        self.blacklist: Set[str] = set()
        
    def generate_token(self,
                      user_id: str,
                      claims: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate JWT tokens
        
        Args:
            user_id: User identifier
            claims: Optional additional claims
            
        Returns:
            Dictionary with access and refresh tokens
        """
        try:
            now = datetime.utcnow()
            
            # Access token
            access_payload = {
                'user_id': user_id,
                'type': 'access',
                'exp': now + timedelta(seconds=self.token_expiry),
                'iat': now
            }
            if claims:
                access_payload.update(claims)
                
            access_token = jwt.encode(
                access_payload,
                self.secret_key,
                algorithm='HS256'
            )
            
            # Refresh token
            refresh_payload = {
                'user_id': user_id,
                'type': 'refresh',
                'exp': now + timedelta(seconds=self.refresh_expiry),
                'iat': now
            }
            refresh_token = jwt.encode(
                refresh_payload,
                self.secret_key,
                algorithm='HS256'
            )
            
            return {
                'access_token': access_token,
                'refresh_token': refresh_token
            }
            
        except Exception as e:
            raise SecurityError(f"Token generation failed: {str(e)}")
            
    def verify_token(self,
                    token: str,
                    token_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify JWT token
        
        Args:
            token: Token to verify
            token_type: Optional token type to check
            
        Returns:
            Token payload
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            if token in self.blacklist:
                raise AuthenticationError("Token has been revoked")
                
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=['HS256']
            )
            
            if token_type and payload.get('type') != token_type:
                raise AuthenticationError(f"Invalid token type: {payload.get('type')}")
                
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
            
    def refresh_token(self, refresh_token: str) -> Dict[str, str]:
        """
        Generate new access token from refresh token
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            Dictionary with new access token
        """
        payload = self.verify_token(refresh_token, 'refresh')
        return self.generate_token(
            payload['user_id'],
            {k: v for k, v in payload.items() if k not in ['exp', 'iat', 'type']}
        )
        
    def revoke_token(self, token: str) -> None:
        """Add token to blacklist"""
        self.blacklist.add(token)

class Permission:
    """Permission definition"""
    
    def __init__(self,
                 name: str,
                 description: Optional[str] = None):
        self.name = name
        self.description = description

class Role:
    """Role definition"""
    
    def __init__(self,
                 name: str,
                 permissions: List[str],
                 description: Optional[str] = None):
        self.name = name
        self.permissions = set(permissions)
        self.description = description

class AccessControl:
    """Access control manager"""
    
    def __init__(self):
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        
    def add_permission(self,
                      name: str,
                      description: Optional[str] = None) -> None:
        """Add permission"""
        self.permissions[name] = Permission(name, description)
        
    def add_role(self,
                 name: str,
                 permissions: List[str],
                 description: Optional[str] = None) -> None:
        """Add role"""
        # Verify permissions exist
        for perm in permissions:
            if perm not in self.permissions:
                raise SecurityError(f"Permission not found: {perm}")
                
        self.roles[name] = Role(name, permissions, description)
        
    def assign_role(self,
                   user_id: str,
                   role_name: str) -> None:
        """Assign role to user"""
        if role_name not in self.roles:
            raise SecurityError(f"Role not found: {role_name}")
            
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
            
        self.user_roles[user_id].add(role_name)
        
    def remove_role(self,
                   user_id: str,
                   role_name: str) -> None:
        """Remove role from user"""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)
            
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for user"""
        permissions = set()
        
        if user_id in self.user_roles:
            for role_name in self.user_roles[user_id]:
                role = self.roles[role_name]
                permissions.update(role.permissions)
                
        return permissions
        
    def check_permission(self,
                        user_id: str,
                        permission: str) -> bool:
        """Check if user has permission"""
        return permission in self.get_user_permissions(user_id)
        
    def require_permission(self,
                         user_id: str,
                         permission: str) -> None:
        """
        Require permission
        
        Raises:
            AuthorizationError: If user lacks permission
        """
        if not self.check_permission(user_id, permission):
            raise AuthorizationError(
                f"Missing required permission: {permission}"
            )

class SecurityPolicy:
    """Security policy configuration"""
    
    def __init__(self):
        # Password requirements
        self.min_password_length = 8
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_numbers = True
        self.require_special = True
        self.max_password_age = 90  # days
        
        # Authentication
        self.max_login_attempts = 5
        self.lockout_duration = 300  # seconds
        self.session_timeout = 3600  # seconds
        
        # Token settings
        self.token_expiry = 3600  # seconds
        self.refresh_expiry = 86400 * 7  # seconds
        
        # Rate limiting
        self.login_rate_limit = 5  # per minute
        self.api_rate_limit = 100  # per minute
        
    def validate_password(self, password: str) -> None:
        """
        Validate password against policy
        
        Raises:
            SecurityError: If password is invalid
        """
        if len(password) < self.min_password_length:
            raise SecurityError(
                f"Password must be at least {self.min_password_length} characters"
            )
            
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            raise SecurityError("Password must contain uppercase letters")
            
        if self.require_lowercase and not re.search(r'[a-z]', password):
            raise SecurityError("Password must contain lowercase letters")
            
        if self.require_numbers and not re.search(r'\d', password):
            raise SecurityError("Password must contain numbers")
            
        if self.require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise SecurityError("Password must contain special characters")

# Global instances
password_hasher = PasswordHasher()
token_manager = TokenManager()
access_control = AccessControl()
security_policy = SecurityPolicy()
