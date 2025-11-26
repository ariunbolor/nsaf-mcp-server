# External Dependencies for NSAF

This directory contains external dependencies and integrations copied from the aiaas project.

## Structure

### `/foundation_models/skills/`
- **OpenAI Integration**: `content_creation_skill.py` - Direct OpenAI API integration
- **Content Generation**: Platform-specific content creation (Facebook, Instagram, Twitter, LinkedIn)
- **API Key Requirements**: Requires `openai_api_key` credential

### `/config/`
- **Configuration Management**: `config_loader.py` - Handles YAML configs and environment variables
- **Default Config**: `default_config.yaml` - Contains model provider settings (OpenAI, Gemini)
- **Environment Templates**: `.env.template` - Environment variable template

### `/api/`
- **REST API Framework**: FastAPI-based web services
- **Authentication**: JWT and API key authentication
- **WebSocket Support**: Real-time communication handlers
- **Marketplace API**: Agent marketplace functionality

### `/utils/`
- **Database Utilities**: Configuration helpers for multiple database types
- **Helper Functions**: Common utility functions

## External Dependencies Identified

### Foundation Model APIs
- **OpenAI**: GPT-4, GPT-3.5-turbo (configured in skills and config)
- **Google Gemini**: Gemini-pro (configured but not implemented)
- **Anthropic**: Not directly integrated

### Database Systems
- **SQLAlchemy**: ORM for multiple database types
- **PostgreSQL**: Production database option
- **Redis**: Caching and session storage
- **SQLite**: Development/testing database

### Web Framework
- **FastAPI**: Web API framework
- **Uvicorn**: ASGI server
- **WebSockets**: Real-time communication
- **Pydantic**: Data validation

### Authentication & Security
- **JWT**: JSON Web Tokens
- **Passlib**: Password hashing
- **Cryptography**: Encryption utilities

### Quantum Computing
- **Qiskit**: IBM Quantum framework (version 1.0.1)

### Additional Services
- **Redis**: In-memory data structure store
- **HTTP Clients**: aiohttp, httpx for external API calls

## Required Environment Variables
```
AGENT_BUILDER_MODEL_PROVIDER_OPENAI_API_KEY=your_openai_key
AGENT_BUILDER_MODEL_PROVIDER_GEMINI_API_KEY=your_gemini_key
AGENT_BUILDER_SECURITY_JWT_SECRET_KEY=your_jwt_secret
AGENT_BUILDER_SECURITY_API_KEY_KEY=your_api_key
AGENT_BUILDER_DATABASE_POSTGRESQL_PASSWORD=your_db_password
```

## Installation Requirements
See `ai_agent_builder_requirements.txt` for complete dependency list including:
- FastAPI ecosystem
- Database drivers
- Quantum computing libraries
- Machine learning libraries
- Testing frameworks