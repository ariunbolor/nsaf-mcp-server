"""
Foundation Models Integration for NSAF
=====================================

Author: Bolorerdene Bundgaa
Contact: bolor@ariunbolor.org
Website: https://bolor.me

Integrates various foundation models (OpenAI, Anthropic, Google) with the NSAF framework.
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

import httpx
import numpy as np
from openai import OpenAI


class ModelProvider(Enum):
    """Supported foundation model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    GOOGLE = "google"
    DEEPSEEK = "deepseek"


@dataclass
class ModelResponse:
    """Response from a foundation model."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    embeddings: Optional[np.ndarray] = None


class FoundationModelClient:
    """
    Unified client for interacting with various foundation models.
    Integrates with external API configurations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the foundation model client.
        
        Args:
            config: Configuration dictionary from NSAF config
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients for available providers."""
        foundation_models = self.config.get("foundation_models", {})
        
        # OpenAI Client
        if "openai" in foundation_models:
            openai_config = foundation_models["openai"]
            api_key = os.getenv("OPENAI_API_KEY") or openai_config.get("api_key")
            if api_key:
                self.clients[ModelProvider.OPENAI] = OpenAI(api_key=api_key)
                self.logger.info("Initialized OpenAI client")
        
        # HTTP client for other APIs
        self.http_client = httpx.AsyncClient()
    
    async def generate_text(self, 
                          prompt: str,
                          provider: ModelProvider = ModelProvider.OPENAI,
                          **kwargs) -> ModelResponse:
        """
        Generate text using the specified foundation model.
        
        Args:
            prompt: Input prompt
            provider: Model provider to use
            **kwargs: Additional model parameters
            
        Returns:
            ModelResponse with generated text
        """
        try:
            if provider == ModelProvider.OPENAI:
                return await self._generate_openai(prompt, **kwargs)
            elif provider == ModelProvider.ANTHROPIC:
                return await self._generate_anthropic(prompt, **kwargs)
            elif provider == ModelProvider.GOOGLE:
                return await self._generate_google(prompt, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise
    
    async def generate_embeddings(self,
                                text: Union[str, List[str]],
                                provider: ModelProvider = ModelProvider.OPENAI) -> np.ndarray:
        """
        Generate embeddings for text using foundation models.
        
        Args:
            text: Text or list of texts to embed
            provider: Model provider to use
            
        Returns:
            Numpy array of embeddings
        """
        try:
            if provider == ModelProvider.OPENAI:
                return await self._embed_openai(text)
            else:
                raise ValueError(f"Embedding not supported for provider: {provider}")
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            # Return random embeddings as fallback
            if isinstance(text, str):
                return np.random.normal(0, 1, (1, 1536))
            else:
                return np.random.normal(0, 1, (len(text), 1536))
    
    async def _generate_openai(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate text using OpenAI API."""
        if ModelProvider.OPENAI not in self.clients:
            raise ValueError("OpenAI client not initialized")
            
        client = self.clients[ModelProvider.OPENAI]
        model_config = self.config["foundation_models"]["openai"]
        
        response = client.chat.completions.create(
            model=kwargs.get("model", model_config.get("model", "gpt-4")),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", model_config.get("temperature", 0.7)),
            max_tokens=kwargs.get("max_tokens", model_config.get("max_tokens", 2048))
        )
        
        return ModelResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider="openai",
            usage=response.usage.model_dump() if response.usage else None
        )
    
    async def _generate_anthropic(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate text using Anthropic API."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
            
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": kwargs.get("model", "claude-3-sonnet-20240229"),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = await self.http_client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        
        result = response.json()
        return ModelResponse(
            content=result["content"][0]["text"],
            model=result["model"],
            provider="anthropic",
            usage=result.get("usage")
        )
    
    async def _generate_google(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate text using Google Gemini API."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
            
        headers = {"Content-Type": "application/json"}
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 2048)
            }
        }
        
        model = kwargs.get("model", "gemini-pro")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        response = await self.http_client.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        content = result["candidates"][0]["content"]["parts"][0]["text"]
        
        return ModelResponse(
            content=content,
            model=model,
            provider="google"
        )
    
    async def _embed_openai(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        if ModelProvider.OPENAI not in self.clients:
            raise ValueError("OpenAI client not initialized")
            
        client = self.clients[ModelProvider.OPENAI]
        
        # Ensure text is a list
        texts = [text] if isinstance(text, str) else text
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        
        embeddings = np.array([data.embedding for data in response.data])
        return embeddings[0] if isinstance(text, str) else embeddings
    
    async def analyze_task_requirements(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze task requirements using foundation models.
        
        Args:
            task: Task dictionary from NSAF framework
            
        Returns:
            Analysis results with suggestions and embeddings
        """
        description = task.get("description", "")
        goals = task.get("goals", [])
        constraints = task.get("constraints", [])
        
        # Create analysis prompt
        prompt = f"""
        Analyze this task and provide insights for optimization:
        
        Task: {description}
        Goals: {[g.get('type', 'unknown') + ': ' + str(g.get('target', 'N/A')) for g in goals]}
        Constraints: {[c.get('type', 'unknown') + ': ' + str(c.get('limit', 'N/A')) for c in constraints]}
        
        Provide:
        1. Task complexity assessment (1-10)
        2. Recommended approach
        3. Potential challenges
        4. Optimization strategies
        
        Format as JSON.
        """
        
        try:
            # Generate analysis
            response = await self.generate_text(prompt)
            
            # Generate task embedding
            embedding = await self.generate_embeddings(description)
            
            return {
                "analysis": response.content,
                "embedding": embedding,
                "model_used": response.model,
                "provider": response.provider
            }
        except Exception as e:
            self.logger.error(f"Task analysis failed: {e}")
            return {
                "analysis": f"Analysis failed: {str(e)}",
                "embedding": np.random.normal(0, 1, (1536,)),
                "model_used": "fallback",
                "provider": "none"
            }
    
    async def optimize_agent_architecture(self, 
                                        requirements: Dict[str, Any],
                                        current_architecture: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use foundation models to optimize agent architecture.
        
        Args:
            requirements: Task requirements
            current_architecture: Current agent architecture
            
        Returns:
            Optimized architecture suggestions
        """
        prompt = f"""
        Optimize this neural agent architecture for the given requirements:
        
        Requirements: {requirements}
        Current Architecture: {current_architecture}
        
        Suggest improvements for:
        1. Layer configuration
        2. Activation functions
        3. Optimization strategy
        4. Training parameters
        
        Provide specific recommendations with reasoning.
        """
        
        try:
            response = await self.generate_text(prompt)
            return {
                "suggestions": response.content,
                "model_used": response.model,
                "provider": response.provider
            }
        except Exception as e:
            self.logger.error(f"Architecture optimization failed: {e}")
            return {
                "suggestions": f"Optimization failed: {str(e)}",
                "model_used": "fallback",
                "provider": "none"
            }
    
    async def close(self):
        """Close HTTP connections."""
        await self.http_client.aclose()


class NSAFFoundationModels:
    """
    Integration layer between NSAF framework and foundation models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with NSAF configuration."""
        self.config = config
        self.client = FoundationModelClient(config)
        self.logger = logging.getLogger(__name__)
    
    async def compute_task_embedding(self, task: Dict[str, Any]) -> np.ndarray:
        """
        Compute embeddings for task clustering module.
        
        Args:
            task: Task dictionary
            
        Returns:
            Task embedding vector
        """
        description = task.get("description", "")
        if not description:
            return np.random.normal(0, 1, (1536,))
            
        return await self.client.generate_embeddings(description)
    
    async def enhance_memory_content(self, memory_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance memory content using foundation models.
        
        Args:
            memory_content: Memory node content
            
        Returns:
            Enhanced memory with additional insights
        """
        content_str = str(memory_content)
        
        prompt = f"""
        Enhance this memory content with additional insights and connections:
        
        Content: {content_str}
        
        Add:
        1. Key concepts
        2. Related domains
        3. Potential applications
        4. Connection hints
        
        Return enhanced JSON structure.
        """
        
        try:
            response = await self.client.generate_text(prompt)
            enhanced = memory_content.copy()
            enhanced["ai_insights"] = response.content
            enhanced["enhancement_model"] = response.model
            return enhanced
        except Exception as e:
            self.logger.error(f"Memory enhancement failed: {e}")
            return memory_content
    
    async def project_intent_outcomes(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Project potential outcomes for intent planning.
        
        Args:
            intent: Intent description and context
            
        Returns:
            List of projected outcomes
        """
        prompt = f"""
        Project potential outcomes and action sequences for this intent:
        
        Intent: {intent.get('description', '')}
        Context: {intent.get('context', {})}
        
        Generate 3-5 possible outcome paths with:
        1. Action sequence
        2. Success probability
        3. Required resources
        4. Risk factors
        
        Format as JSON array.
        """
        
        try:
            response = await self.client.generate_text(prompt)
            return [{
                "projections": response.content,
                "model_used": response.model,
                "confidence": 0.8  # Default confidence
            }]
        except Exception as e:
            self.logger.error(f"Intent projection failed: {e}")
            return [{
                "projections": f"Projection failed: {str(e)}",
                "model_used": "fallback",
                "confidence": 0.1
            }]
    
    async def close(self):
        """Cleanup resources."""
        await self.client.close()