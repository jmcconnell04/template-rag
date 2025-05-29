# server/app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field # Import Field for accessing default
from typing import Optional

class Settings(BaseSettings):
    # --- General Workspace & Server Config ---
    WORKSPACE_NAME: str = Field(default="rag_poc") # Define default using Field
    RAG_API_HOST_PORT: int = 8000 
    LOG_LEVEL: str = "INFO"

    # --- OpenWebUI Admin Credentials (primarily for docker-compose to pass to OWI) ---
    WEBUI_ADMIN_USER_EMAIL: Optional[str] = "admin@example.com"
    WEBUI_ADMIN_USER_PASSWORD: Optional[str] = "changeme"

    # --- Ollama Configuration ---
    OLLAMA_BASE_URL: str = "http://ollama:11434"

    # --- Database Configuration ---
    DB_TYPE: str = "sqlite" 

    SQLITE_DB_FILE_NAME: str = "app_data.db"

    # PostgreSQL specific
    POSTGRES_HOST: str = "postgres_db"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "rag_user"
    POSTGRES_PASSWORD: str = "rag_password"
    # Set a simple default here, will be updated after WORKSPACE_NAME is loaded
    POSTGRES_DB: str = "ragdb_default" 

    # SQL Server specific
    SQLSERVER_HOST: str = "sqlserver_db"
    SQLSERVER_PORT: int = 1433
    SQLSERVER_USER: str = "sa"
    SQLSERVER_SA_PASSWORD: str = "YourStrongPassword123!"
    # Set a simple default here, will be updated after WORKSPACE_NAME is loaded
    SQLSERVER_DB_NAME: str = "ragdb_default" 
    SQLSERVER_ODBC_DRIVER: Optional[str] = None

    # --- RAG / ChromaDB / Embedding Configuration ---
    CHROMA_PERSIST_DIRECTORY: str = "/app/chroma_data"
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RAG_TOP_K: int = 3

    # --- API Keys for Cloud LLMs ---
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    # --- Auto-Seeding Configuration ---
    AUTO_SEED_ON_STARTUP: bool = True
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore', case_sensitive=False)

# Instantiate settings first
settings = Settings()

# Now, update DB names based on the loaded WORKSPACE_NAME (from .env or default)
# This ensures that if WORKSPACE_NAME is set by .env, it's used.
# If it used its Pydantic default, that's used.
if settings.POSTGRES_DB == "ragdb_default": # Check if it's still the initial simple default
    settings.POSTGRES_DB = f"ragdb_{settings.WORKSPACE_NAME}"

if settings.SQLSERVER_DB_NAME == "ragdb_default": # Check if it's still the initial simple default
    settings.SQLSERVER_DB_NAME = f"ragdb_{settings.WORKSPACE_NAME}"

