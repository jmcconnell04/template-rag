# server/app/db/database.py
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError
import time
import urllib.parse # For SQL Server password escaping
import logging # Import logging

from ..config import settings

logger = logging.getLogger(__name__) # Get a logger instance for this module

SQLALCHEMY_DATABASE_URL = ""
engine_args = {}

if settings.DB_TYPE.lower() == "sqlite":
    db_path_in_container = f"/app/database/{settings.SQLITE_DB_FILE_NAME}"
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{db_path_in_container}"
    engine_args["connect_args"] = {"check_same_thread": False}
    logger.info(f"Configuring SQLite database at: {db_path_in_container}")

elif settings.DB_TYPE.lower() == "postgres":
    SQLALCHEMY_DATABASE_URL = (
        f"postgresql+psycopg2://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )
    engine_args["pool_pre_ping"] = True
    logger.info(f"Configuring PostgreSQL: {settings.POSTGRES_HOST}/{settings.POSTGRES_DB}")

elif settings.DB_TYPE.lower() == "sqlserver":
    sa_password_encoded = urllib.parse.quote_plus(settings.SQLSERVER_SA_PASSWORD)
    conn_str_base = (
        f"mssql+pyodbc://{settings.SQLSERVER_USER}:{sa_password_encoded}"
        f"@{settings.SQLSERVER_HOST}:{settings.SQLSERVER_PORT}/{settings.SQLSERVER_DB_NAME}"
    )
    driver_part = ""
    if settings.SQLSERVER_ODBC_DRIVER:
        driver = settings.SQLSERVER_ODBC_DRIVER.replace(' ', '+')
        driver_part = f"driver={driver}"
    else:
        # Attempt common default drivers for Linux containers.
        # This might need adjustment based on what's in the Docker image.
        # If multiple are present, pyodbc might pick one, or it might fail.
        # Explicitly setting SQLSERVER_ODBC_DRIVER in .env is more reliable.
        possible_drivers = ["ODBC+Driver+18+for+SQL+Server", "ODBC+Driver+17+for+SQL+Server"]
        # For simplicity, we'll let SQLAlchemy try to auto-detect or use a common one.
        # If issues, user should set SQLSERVER_ODBC_DRIVER in .env
        # Example: driver_part = "driver=ODBC+Driver+18+for+SQL+Server"
        logger.warning("SQLSERVER_ODBC_DRIVER not specified in .env. Relying on auto-detection or common defaults. "
                       "If connection fails, please specify the correct driver.")
        # Defaulting to 18 for now if none specified
        driver_part = "driver=ODBC+Driver+18+for+SQL+Server"


    # For development with Dockerized SQL Server, TrustServerCertificate may be needed.
    # For production, proper certificate validation is essential.
    # Ensure there's a '?' if driver_part is not empty, otherwise add it.
    separator = "?" if "?" not in conn_str_base else "&"
    if driver_part:
         conn_str = f"{conn_str_base}{separator}{driver_part}&TrustServerCertificate=yes"
    else: # If no driver specified, just add TrustServerCertificate
         conn_str = f"{conn_str_base}{separator}TrustServerCertificate=yes"
    
    SQLALCHEMY_DATABASE_URL = conn_str
    engine_args["pool_pre_ping"] = True
    logger.info(f"Configuring SQL Server: {settings.SQLSERVER_HOST}/{settings.SQLSERVER_DB_NAME}")
    logger.debug(f"SQL Server Connection String (password masked): {engine_args.get('url', SQLALCHEMY_DATABASE_URL).replace(sa_password_encoded, '********')}")


else:
    err_msg = f"Unsupported DB_TYPE: '{settings.DB_TYPE}'. Choose 'sqlite', 'postgres', or 'sqlserver'."
    logger.critical(err_msg) # Use critical for fatal config errors
    raise ValueError(err_msg)

engine = create_engine(SQLALCHEMY_DATABASE_URL, **engine_args)

if settings.DB_TYPE.lower() == "sqlite":
    @event.listens_for(Engine, "connect")
    def set_sqlite_pragmas(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA busy_timeout = 5000")
            cursor.execute("PRAGMA journal_mode=WAL;")
            logger.info("SQLite PRAGMA busy_timeout=5000ms and journal_mode=WAL set for new connection.")
        except Exception as e:
            logger.warning(f"Failed to set SQLite PRAGMAs: {e}")
        finally:
            cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def _wait_for_db(db_engine, db_type_name_for_log):
    max_retries = 12 # Increased retries for slower DB starts
    retry_interval = 5  # seconds
    logger.info(f"Attempting to connect to {db_type_name_for_log} database...")
    for attempt in range(max_retries):
        try:
            with db_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            logger.info(f"{db_type_name_for_log} database connection successful.")
            return True
        except SQLAlchemyOperationalError as e:
            logger.warning(f"{db_type_name_for_log} connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_interval}s...")
            if attempt < max_retries - 1:
                time.sleep(retry_interval)
            else:
                logger.error(f"{db_type_name_for_log} database connection failed after {max_retries} retries.")
                return False
        except Exception as e: # Catch other potential errors like driver issues
             logger.error(f"Unexpected error connecting to {db_type_name_for_log}: {e}")
             import traceback
             logger.error(traceback.format_exc())
             return False

def create_db_and_tables():
    logger.info(f"Target Database Type: {settings.DB_TYPE.upper()}")
    logger.info(f"Target Connection URL (password masked): {engine.url.render_as_string(hide_password=True)}")

    if settings.DB_TYPE.lower() in ["postgres", "sqlserver"]:
        if not _wait_for_db(engine, settings.DB_TYPE.upper()):
            # Log critical error but allow app to continue starting; individual operations will fail.
            # Or, raise RuntimeError to halt app startup if DB is absolutely essential before app can run.
            logger.critical(f"Could not connect to {settings.DB_TYPE.upper()} database during startup. Application might not function correctly.")
            # raise RuntimeError(f"Failed to connect to {settings.DB_TYPE.upper()} database after multiple retries.")
    
    try:
        logger.info("Attempting to create database tables if they don't exist...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables checked/created successfully.")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # This could be a critical failure, consider raising an exception

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

