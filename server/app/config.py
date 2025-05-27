from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # General Workspace Settings (from .env)
    WORKSPACE_NAME: str = "rag_poc" 

    # Server Configuration (from .env)
    RAG_API_HOST_PORT: int = 8000 

    # Ollama Configuration (from .env)
    OLLAMA_BASE_URL: str = "http://ollama:11434"

    # OpenWebUI Admin Credentials (from .env, if server needs to read them)
    # Add these if your server logic needs to know the admin email
    # Make them Optional if they might not always be set in .env for the server
    WEBUI_ADMIN_USER_EMAIL: Optional[str] = None 
    WEBUI_ADMIN_USER_PASSWORD: Optional[str] = None 

    # API Keys (will be used later)
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    # Seeding configuration
    AUTO_SEED_ON_STARTUP: bool = True

    # For loading from .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

settings = Settings()