# server/app/db/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# The database file will be created in the '/app/database/' directory inside the container,
# which is bind-mounted from './rag_files/metadata_db/' on the host.
DATABASE_FILE_NAME = "app_data.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:////app/database/{DATABASE_FILE_NAME}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}  # Required for SQLite with FastAPI/multi-threading
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Function to create all tables defined that inherit from Base
def create_db_and_tables():
    Base.metadata.create_all(bind=engine)

# Dependency for FastAPI to get a DB session for each request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()