# server/app/main.py
from fastapi import FastAPI

# Initialize FastAPI app
# You can add title, version, etc. later as needed
app = FastAPI(title="RAG Workspace Server")

@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Simple health check endpoint to confirm the server is running and responsive.
    """
    return {"status": "ok", "message": "RAG Server is healthy"}

# You can add more routes and application logic below later.
# For example, to see the dynamic WORKSPACE_NAME from .env (optional for this step):
# from .config import settings # We'll create config.py in a later step
# @app.get("/", tags=["Root"])
# async def read_root():
#     return {"message": f"Welcome to {settings.WORKSPACE_NAME} RAG Server"}