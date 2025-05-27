# server/app/logger_config.py
import logging
import sys
from ..config import settings # To potentially use WORKSPACE_NAME or log level from config

def setup_logging():
    # More advanced logging configuration can go here
    # For now, a basic setup that logs to stdout, which Docker will capture
    log_format = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - {settings.WORKSPACE_NAME} - %(message)s"

    logging.basicConfig(
        level=logging.INFO, # Or load from settings.LOG_LEVEL
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout) # Log to stdout
        ]
    )
    # You can also add file handlers here if needed

    # Silence overly verbose loggers if necessary
    # logging.getLogger("httpx").setLevel(logging.WARNING)
    # logging.getLogger("httpcore").setLevel(logging.WARNING)
    # logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.WARNING) # Chroma's telemetry
    # logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.WARNING)


# Call setup_logging once when the module is imported,
# or explicitly call it in main.py startup.
# For simplicity in a FastAPI app, often it's called once when app starts,
# or Uvicorn's logger is configured.
# Let's make it a function to be called from main.py startup for clarity.