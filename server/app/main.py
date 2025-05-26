# server/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx # For making async HTTP calls
import json # For parsing Ollama's potentially multi-JSON responses if streamed (though not for this first test)

# Import settings from our new config.py
from .config import settings

app = FastAPI(title="RAG Workspace Server")

# Create an asynchronous HTTP client that can be reused across requests.
# It's good practice to manage its lifecycle with startup/shutdown events.
http_client = httpx.AsyncClient()

@app.on_event("startup")
async def startup_event():
    # This function will be called when the FastAPI application starts.
    # You can initialize resources here, like our HTTP client.
    global http_client
    if http_client.is_closed: # Re-initialize if it was closed (e.g., after tests)
        http_client = httpx.AsyncClient()
    print(f"RAG Server for {settings.WORKSPACE_NAME} started. Ollama URL: {settings.OLLAMA_BASE_URL}")

@app.on_event("shutdown")
async def shutdown_event():
    # This function will be called when the FastAPI application shuts down.
    # Clean up resources here, like closing the HTTP client.
    await http_client.aclose()
    print(f"RAG Server for {settings.WORKSPACE_NAME} shutting down.")


@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok", "message": f"RAG Server for {settings.WORKSPACE_NAME} is healthy"}

# Pydantic model for the request body of our new Ollama prompt endpoint
class OllamaSimplePromptRequest(BaseModel):
    model: str = "gemma:2b"  # Default model, ensure it's pulled in your Ollama instance
    prompt: str
    stream: bool = False     # For this simple test, we'll use non-streaming

@app.post("/test_ollama_prompt", tags=["Ollama Tests"])
async def test_ollama_prompt(request_data: OllamaSimplePromptRequest):
    """
    Sends a prompt to the configured Ollama instance's /api/generate endpoint
    and returns the (non-streamed) response.
    """
    if not settings.OLLAMA_BASE_URL:
        raise HTTPException(status_code=500, detail="OLLAMA_BASE_URL is not configured in settings.")

    ollama_generate_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    
    payload = {
        "model": request_data.model,
        "prompt": request_data.prompt,
        "stream": request_data.stream
    }

    try:
        # Make the POST request to Ollama
        # Increased timeout as model generation can take time
        response = await http_client.post(ollama_generate_url, json=payload, timeout=120.0)
        
        # Raise an exception for HTTP errors (4xx or 5xx)
        response.raise_for_status()
        
        # Parse the JSON response from Ollama
        ollama_response = response.json()
        
        return {
            "model_used": request_data.model,
            "ollama_response": ollama_response.get("response", ""), # The generated text
            "full_ollama_payload": ollama_response # For debugging, see the whole thing
        }

    except httpx.HTTPStatusError as e:
        # More detailed error for issues like model not found, etc.
        error_detail = f"Ollama API Error: {e.response.status_code} - Response: {e.response.text}"
        print(error_detail) # Log to server console
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        # For network issues, timeout, etc.
        error_detail = f"Request error connecting to Ollama ({ollama_generate_url}): {str(e)}"
        print(error_detail) # Log to server console
        raise HTTPException(status_code=503, detail=error_detail) # 503 Service Unavailable
    except json.JSONDecodeError as e:
        error_detail = f"Failed to decode JSON response from Ollama. Status: {response.status_code}, Response: {response.text}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)


# Optional: Update root endpoint to show loaded settings
@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": f"Welcome to {settings.WORKSPACE_NAME} RAG Server!",
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "auto_seed_on_startup": settings.AUTO_SEED_ON_STARTUP
    }