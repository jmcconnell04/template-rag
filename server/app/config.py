# server/app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # General Workspace Settings (from .env)
    WORKSPACE_NAME: str = "rag_poc" # Default if not in .env

    # Server Configuration (from .env)
    RAG_API_HOST_PORT: int = 8000 # Default if not in .env, for Docker host mapping

    # Ollama Configuration (from .env)
    OLLAMA_BASE_URL: str = "http://ollama:11434" # Default internal Docker URL

    # API Keys (will be used later, good to define now)
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    # Seeding configuration
    AUTO_SEED_ON_STARTUP: bool = True

    # For loading from .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

settings = Settings()