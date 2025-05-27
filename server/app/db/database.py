# server/app/db/database.py
from sqlalchemy import create_engine, event # Add event
from sqlalchemy.engine import Engine      # Add Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import logging

logger = logging.getLogger(__name__)

DATABASE_FILE_NAME = "app_data.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:////app/database/{DATABASE_FILE_NAME}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Set PRAGMA busy_timeout on new connections to SQLite
@event.listens_for(Engine, "connect")
def set_sqlite_busy_timeout(dbapi_connection, connection_record):
    # This event is triggered each time SQLAlchemy makes a new DBAPI connection.
    # For SQLite, dbapi_connection is a sqlite3.Connection object.
    cursor = dbapi_connection.cursor()
    try:
        # Set timeout to 5000 milliseconds (5 seconds)
        cursor.execute("PRAGMA busy_timeout = 5000")
        logger.info("SQLite PRAGMA busy_timeout set to 5000ms for new connection.")
    except Exception as e:
        # Log if setting pragma fails, though it's unlikely for busy_timeout
        logger.warning(f"Failed to set PRAGMA busy_timeout: {e}")
    finally:
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def create_db_and_tables():
    logger.info(f"Database URL: {SQLALCHEMY_DATABASE_URL}")
    logger.info("Initializing database and creating tables if they don't exist...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables checked/created.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()