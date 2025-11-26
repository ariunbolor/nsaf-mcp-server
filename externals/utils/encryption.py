"""
Encryption Module
--------------
Encryption utilities for the agent builder system.
"""

from typing import Union, Optional, Dict, Any
import base64
import os
from pathlib import Path
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

from .exceptions import EncryptionError
from .logging import logger

class KeyManager:
    """Key management utility"""
    
    def __init__(self,
                 key_file: Optional[Union[str, Path]] = None,
                 iterations: int = 100000):
        self.key_file = Path(key_file) if key_file else None
        self.iterations = iterations
        self.keys: Dict[str, bytes] = {}
        
        if self.key_file and self.key_file.exists():
            self._load_keys()
            
    def _load_keys(self) -> None:
        """Load keys from file"""
        try:
            with open(self.key_file, 'rb') as f:
                encoded = f.read()
                data = json.loads(encoded.decode())
                
                self.keys = {
                    key: base64.b64decode(value)
                    for key, value in data.items()
                }
        except Exception as e:
            raise EncryptionError(f"Failed to load keys: {str(e)}")
            
    def _save_keys(self) -> None:
        """Save keys to file"""
        if not self.key_file:
            return
            
        try:
            # Ensure directory exists
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                key: base64.b64encode(value).decode()
                for key, value in self.keys.items()
            }
            
            with open(self.key_file, 'wb') as f:
                f.write(json.dumps(data).encode())
                
        except Exception as e:
            raise EncryptionError(f"Failed to save keys: {str(e)}")
            
    def generate_key(self,
                    key_id: str,
                    password: Optional[str] = None,
                    salt: Optional[bytes] = None) -> bytes:
        """
        Generate encryption key
        
        Args:
            key_id: Key identifier
            password: Optional password for key derivation
            salt: Optional salt for key derivation
            
        Returns:
            Generated key
        """
        try:
            if password:
                # Derive key from password
                salt = salt or os.urandom(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=self.iterations,
                    backend=default_backend()
                )
                key = kdf.derive(password.encode())
            else:
                # Generate random key
                key = Fernet.generate_key()
                
            self.keys[key_id] = key
            self._save_keys()
            return key
            
        except Exception as e:
            raise EncryptionError(f"Failed to generate key: {str(e)}")
            
    def get_key(self, key_id: str) -> bytes:
        """Get encryption key"""
        if key_id not in self.keys:
            raise EncryptionError(f"Key not found: {key_id}")
        return self.keys[key_id]
        
    def delete_key(self, key_id: str) -> None:
        """Delete encryption key"""
        if key_id in self.keys:
            del self.keys[key_id]
            self._save_keys()

class Encryptor:
    """Encryption utility"""
    
    def __init__(self, key: bytes):
        self.fernet = Fernet(key)
        
    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data
        
        Args:
            data: Data to encrypt
            
        Returns:
            Base64 encoded encrypted data
        """
        try:
            if isinstance(data, str):
                data = data.encode()
                
            encrypted = self.fernet.encrypt(data)
            return base64.b64encode(encrypted).decode()
            
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {str(e)}")
            
    def decrypt(self, data: str) -> bytes:
        """
        Decrypt data
        
        Args:
            data: Base64 encoded encrypted data
            
        Returns:
            Decrypted data
        """
        try:
            encrypted = base64.b64decode(data)
            return self.fernet.decrypt(encrypted)
            
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {str(e)}")

class AESEncryptor:
    """AES encryption utility"""
    
    def __init__(self,
                 key: bytes,
                 mode: str = "CBC"):
        if len(key) not in [16, 24, 32]:  # 128, 192, or 256 bit
            raise EncryptionError("Invalid key length")
            
        self.key = key
        self.mode = mode
        self.backend = default_backend()
        self.padder = padding.PKCS7(128)
        
    def encrypt(self, data: Union[str, bytes]) -> Dict[str, str]:
        """
        Encrypt data using AES
        
        Args:
            data: Data to encrypt
            
        Returns:
            Dictionary with encrypted data and IV
        """
        try:
            if isinstance(data, str):
                data = data.encode()
                
            # Generate IV
            iv = os.urandom(16)
            
            # Create cipher
            if self.mode == "CBC":
                cipher = Cipher(
                    algorithms.AES(self.key),
                    modes.CBC(iv),
                    backend=self.backend
                )
            else:
                raise EncryptionError(f"Unsupported mode: {self.mode}")
                
            # Pad data
            padder = self.padder.padder()
            padded_data = padder.update(data) + padder.finalize()
            
            # Encrypt
            encryptor = cipher.encryptor()
            encrypted = encryptor.update(padded_data) + encryptor.finalize()
            
            return {
                'encrypted': base64.b64encode(encrypted).decode(),
                'iv': base64.b64encode(iv).decode()
            }
            
        except Exception as e:
            raise EncryptionError(f"AES encryption failed: {str(e)}")
            
    def decrypt(self,
               encrypted: str,
               iv: str) -> bytes:
        """
        Decrypt AES encrypted data
        
        Args:
            encrypted: Base64 encoded encrypted data
            iv: Base64 encoded initialization vector
            
        Returns:
            Decrypted data
        """
        try:
            # Decode data
            encrypted_data = base64.b64decode(encrypted)
            iv_bytes = base64.b64decode(iv)
            
            # Create cipher
            if self.mode == "CBC":
                cipher = Cipher(
                    algorithms.AES(self.key),
                    modes.CBC(iv_bytes),
                    backend=self.backend
                )
            else:
                raise EncryptionError(f"Unsupported mode: {self.mode}")
                
            # Decrypt
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            # Unpad
            unpadder = self.padder.unpadder()
            return unpadder.update(padded_data) + unpadder.finalize()
            
        except Exception as e:
            raise EncryptionError(f"AES decryption failed: {str(e)}")

class EncryptedFile:
    """Utility for working with encrypted files"""
    
    def __init__(self,
                 path: Union[str, Path],
                 key: bytes):
        self.path = Path(path)
        self.encryptor = Encryptor(key)
        
    def write(self, data: Union[str, bytes]) -> None:
        """Write encrypted data to file"""
        try:
            # Ensure directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)
            
            # Encrypt data
            encrypted = self.encryptor.encrypt(data)
            
            # Write to file
            with open(self.path, 'w') as f:
                f.write(encrypted)
                
        except Exception as e:
            raise EncryptionError(f"Failed to write encrypted file: {str(e)}")
            
    def read(self) -> bytes:
        """Read encrypted data from file"""
        try:
            # Read from file
            with open(self.path) as f:
                encrypted = f.read()
                
            # Decrypt data
            return self.encryptor.decrypt(encrypted)
            
        except Exception as e:
            raise EncryptionError(f"Failed to read encrypted file: {str(e)}")

# Global key manager instance
key_manager = KeyManager(
    key_file=Path("keys/encryption_keys.json")
)
