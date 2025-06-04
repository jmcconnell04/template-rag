# server/app/logger_config.py
import logging
import sys
# from ..config import settings # OLD - Incorrect relative import
from .config import settings   # CORRECTED: config.py is in the same 'app' package/directory

# Determine log level from settings, defaulting to INFO
LOG_LEVEL_STR = getattr(settings, 'LOG_LEVEL', 'INFO').upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

def setup_logging():
    """
    Configures the root logger for the application.
    Logs to stdout, which will be captured by Docker.
    """
    # Using WORKSPACE_NAME in the log format for better context if multiple instances log to the same place
    log_format = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - [{settings.WORKSPACE_NAME}] - %(message)s"
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Remove any existing handlers to avoid duplicate logs if this is called multiple times
    # or if Uvicorn/FastAPI also tries to set up logging.
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=LOG_LEVEL,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout) # Log to stdout
        ]
    )

    # Optionally, set levels for overly verbose third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.WARNING) 
    logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.WARNING)
    # logging.getLogger("pika").setLevel(logging.WARNING) # If RabbitMQ/Pika is ever used

    logger = logging.getLogger(__name__) # Get logger for this module itself
    logger.info(f"Logging configured with level: {logging.getLevelName(LOG_LEVEL)}")

# Note: setup_logging() should be called once at the application startup,
# for example, in the main.py's startup_event.
