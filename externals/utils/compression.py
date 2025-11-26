"""
Compression Module
--------------
Compression utilities for the agent builder system.
"""

from typing import Union, Optional, Dict, Any
import zlib
import gzip
import bz2
import lzma
import base64
from pathlib import Path
import json

from .exceptions import CompressionError
from .logging import logger

class Compressor:
    """Base compressor class"""
    
    def compress(self, data: Union[str, bytes]) -> bytes:
        """
        Compress data
        
        Args:
            data: Data to compress
            
        Returns:
            Compressed bytes
        """
        raise NotImplementedError
        
    def decompress(self, data: bytes) -> bytes:
        """
        Decompress data
        
        Args:
            data: Data to decompress
            
        Returns:
            Decompressed bytes
        """
        raise NotImplementedError

class ZlibCompressor(Compressor):
    """Zlib compressor"""
    
    def __init__(self, level: int = 6):
        self.level = level
        
    def compress(self, data: Union[str, bytes]) -> bytes:
        """Compress with zlib"""
        try:
            if isinstance(data, str):
                data = data.encode()
                
            return zlib.compress(data, level=self.level)
            
        except Exception as e:
            raise CompressionError(f"Zlib compression failed: {str(e)}")
            
    def decompress(self, data: bytes) -> bytes:
        """Decompress with zlib"""
        try:
            return zlib.decompress(data)
            
        except Exception as e:
            raise CompressionError(f"Zlib decompression failed: {str(e)}")

class GzipCompressor(Compressor):
    """Gzip compressor"""
    
    def __init__(self, level: int = 6):
        self.level = level
        
    def compress(self, data: Union[str, bytes]) -> bytes:
        """Compress with gzip"""
        try:
            if isinstance(data, str):
                data = data.encode()
                
            return gzip.compress(data, compresslevel=self.level)
            
        except Exception as e:
            raise CompressionError(f"Gzip compression failed: {str(e)}")
            
    def decompress(self, data: bytes) -> bytes:
        """Decompress with gzip"""
        try:
            return gzip.decompress(data)
            
        except Exception as e:
            raise CompressionError(f"Gzip decompression failed: {str(e)}")

class Bzip2Compressor(Compressor):
    """Bzip2 compressor"""
    
    def __init__(self, level: int = 9):
        self.level = level
        
    def compress(self, data: Union[str, bytes]) -> bytes:
        """Compress with bzip2"""
        try:
            if isinstance(data, str):
                data = data.encode()
                
            return bz2.compress(data, compresslevel=self.level)
            
        except Exception as e:
            raise CompressionError(f"Bzip2 compression failed: {str(e)}")
            
    def decompress(self, data: bytes) -> bytes:
        """Decompress with bzip2"""
        try:
            return bz2.decompress(data)
            
        except Exception as e:
            raise CompressionError(f"Bzip2 decompression failed: {str(e)}")

class LzmaCompressor(Compressor):
    """LZMA compressor"""
    
    def __init__(self, preset: int = 6):
        self.preset = preset
        
    def compress(self, data: Union[str, bytes]) -> bytes:
        """Compress with LZMA"""
        try:
            if isinstance(data, str):
                data = data.encode()
                
            return lzma.compress(data, preset=self.preset)
            
        except Exception as e:
            raise CompressionError(f"LZMA compression failed: {str(e)}")
            
    def decompress(self, data: bytes) -> bytes:
        """Decompress with LZMA"""
        try:
            return lzma.decompress(data)
            
        except Exception as e:
            raise CompressionError(f"LZMA decompression failed: {str(e)}")

class CompressedData:
    """Container for compressed data"""
    
    def __init__(self,
                 data: bytes,
                 algorithm: str,
                 original_size: int):
        self.data = data
        self.algorithm = algorithm
        self.original_size = original_size
        self.compressed_size = len(data)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'data': base64.b64encode(self.data).decode(),
            'algorithm': self.algorithm,
            'original_size': self.original_size,
            'compressed_size': self.compressed_size
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressedData':
        """Create from dictionary"""
        return cls(
            base64.b64decode(data['data']),
            data['algorithm'],
            data['original_size']
        )

class CompressionManager:
    """Compression manager"""
    
    def __init__(self):
        self.compressors = {
            'zlib': ZlibCompressor(),
            'gzip': GzipCompressor(),
            'bzip2': Bzip2Compressor(),
            'lzma': LzmaCompressor()
        }
        
    def compress(self,
                data: Union[str, bytes],
                algorithm: str = "zlib") -> CompressedData:
        """
        Compress data
        
        Args:
            data: Data to compress
            algorithm: Compression algorithm
            
        Returns:
            CompressedData object
        """
        if algorithm not in self.compressors:
            raise CompressionError(f"Unknown algorithm: {algorithm}")
            
        compressor = self.compressors[algorithm]
        original_size = len(data if isinstance(data, bytes) else data.encode())
        
        compressed = compressor.compress(data)
        return CompressedData(compressed, algorithm, original_size)
        
    def decompress(self, compressed: CompressedData) -> bytes:
        """
        Decompress data
        
        Args:
            compressed: CompressedData object
            
        Returns:
            Decompressed bytes
        """
        if compressed.algorithm not in self.compressors:
            raise CompressionError(
                f"Unknown algorithm: {compressed.algorithm}"
            )
            
        compressor = self.compressors[compressed.algorithm]
        return compressor.decompress(compressed.data)

class CompressedFile:
    """Utility for working with compressed files"""
    
    def __init__(self,
                 path: Union[str, Path],
                 algorithm: str = "zlib"):
        self.path = Path(path)
        self.algorithm = algorithm
        self.manager = CompressionManager()
        
    def write(self, data: Union[str, bytes]) -> None:
        """Write compressed data to file"""
        try:
            # Ensure directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)
            
            # Compress data
            compressed = self.manager.compress(data, self.algorithm)
            
            # Write metadata and compressed data
            with open(self.path, 'w') as f:
                json.dump(compressed.to_dict(), f)
                
        except Exception as e:
            raise CompressionError(
                f"Failed to write compressed file: {str(e)}"
            )
            
    def read(self) -> bytes:
        """Read compressed data from file"""
        try:
            if not self.path.exists():
                raise CompressionError(f"File not found: {self.path}")
                
            # Read metadata and compressed data
            with open(self.path) as f:
                data = json.load(f)
                
            compressed = CompressedData.from_dict(data)
            return self.manager.decompress(compressed)
            
        except Exception as e:
            raise CompressionError(
                f"Failed to read compressed file: {str(e)}"
            )

# Global compression manager instance
compression_manager = CompressionManager()
