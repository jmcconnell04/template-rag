# server/app/main.py
# server/app/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json
import time
import uuid

# Add or ensure 'List' is imported from typing
from typing import List, Dict, Optional, Union # Ensure List is here

# Import settings from our new config.py
from .config import settings
# Assuming openai_models.py is in the same 'app' directory
from .openai_models import (
    ModelCard, ModelList, ChatCompletionRequest, ChatMessageInput, # ChatMessageInput is used here
    ChatCompletionStreamResponse, ChatCompletionStreamChoice, ChatCompletionStreamChoiceDelta,
    ChatCompletionResponse, ChatCompletionChoice, ChatMessageOutput, Usage
)

app = FastAPI(title="RAG Workspace Server")
http_client = httpx.AsyncClient()

@app.on_event("startup")
async def startup_event():
    global http_client
    if http_client.is_closed:
        http_client = httpx.AsyncClient()
    print(f"RAG Server for {settings.WORKSPACE_NAME} started. Ollama URL: {settings.OLLAMA_BASE_URL}")

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
    print(f"RAG Server for {settings.WORKSPACE_NAME} shutting down.")

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok", "message": f"RAG Server for {settings.WORKSPACE_NAME} is healthy"}

@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": f"Welcome to {settings.WORKSPACE_NAME} RAG Server!",
        "ollama_base_url": settings.OLLAMA_BASE_URL,
    }

# === OpenAI Compatible Endpoints (/v1) ===

@app.get("/v1/models", response_model=ModelList, tags=["OpenAI Compatibility"])
async def list_models():
    """
    Provides a list of available models to OpenWebUI.
    For v1, we'll offer the default Ollama model we are testing with.
    Later, this will be dynamic (Ollama, Gemini, OpenAI).
    """
    # TODO: Later, dynamically populate based on available Ollama models and configured cloud models.
    # For now, hardcode the model(s) you've pulled into Ollama and want to test.
    # Ensure the 'id' matches what OpenWebUI will send back in ChatCompletionRequest.
    # The 'owned_by' can be 'ollama', 'openai', 'google', etc.
    available_models = [
        ModelCard(id="ollama/gemma:2b", owned_by="ollama"), # Ensure this model is pulled in your Ollama
        # You can add more models here that you've pulled into Ollama
        # ModelCard(id="ollama/qwen:1.8b", owned_by="ollama"),
    ]
    if settings.GEMINI_API_KEY:
        available_models.append(ModelCard(id="gemini/gemini-1.5-flash-latest", owned_by="google"))
    if settings.OPENAI_API_KEY:
        available_models.append(ModelCard(id="openai/gpt-3.5-turbo", owned_by="openai"))
        
    return ModelList(data=available_models)

async def ollama_chat_stream_generator(model_id: str, messages: List[ChatMessageInput], request_id: str):
    """
    Generator function to stream responses from Ollama's /api/chat endpoint,
    formatted as OpenAI-compatible Server-Sent Events (SSE).
    """
    ollama_model_name = model_id.replace("ollama/", "") # Remove prefix for Ollama
    
    # Convert OpenAI message format to Ollama message format
    ollama_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

    payload = {
        "model": ollama_model_name,
        "messages": ollama_messages,
        "stream": True
    }
    
    try:
        async with http_client.stream("POST", f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=120.0) as response:
            if response.status_code != 200:
                error_content = await response.aread()
                print(f"Ollama API Error ({response.status_code}): {error_content.decode()}")
                # Yield an error in SSE format (though client might not expect it)
                error_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": f"Error: Ollama API Error {response.status_code}"}, "finish_reason": "error"}]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield f"data: [DONE]\n\n" # Still send DONE to close stream gracefully on client
                return

            async for line in response.aiter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        delta_content = chunk.get("message", {}).get("content", "")
                        
                        stream_choice = ChatCompletionStreamChoice(
                            index=0,
                            delta=ChatCompletionStreamChoiceDelta(content=delta_content, role="assistant" if chunk.get("message", {}).get("role") == "assistant" and delta_content else None),
                            finish_reason= "stop" if chunk.get("done") else None
                        )
                        stream_resp = ChatCompletionStreamResponse(
                            id=request_id, # Use the same ID for all chunks of a response
                            model=model_id, 
                            choices=[stream_choice]
                        )
                        yield f"data: {stream_resp.model_dump_json()}\n\n"
                        
                        if chunk.get("done"):
                            break 
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON line from Ollama stream: {line}")
                        continue
        yield f"data: [DONE]\n\n" # Signal end of stream to client
    except httpx.RequestError as e:
        print(f"Request error connecting to Ollama for streaming: {str(e)}")
        # Yield an error message in SSE format
        error_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "delta": {"content": f"Error: Could not connect to Ollama - {str(e)}"}, "finish_reason": "error"}]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield f"data: [DONE]\n\n"
    except Exception as e:
        print(f"Generic error in Ollama stream generator: {str(e)}")
        error_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "delta": {"content": f"Error: An unexpected error occurred - {str(e)}"}, "finish_reason": "error"}]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield f"data: [DONE]\n\n"


@app.post("/v1/chat/completions", tags=["OpenAI Compatibility"])
async def chat_completions(request: ChatCompletionRequest, fastapi_request: FastAPIRequest): # Added FastAPIRequest
    """
    OpenAI-compatible chat completions endpoint.
    Routes to Ollama, and later Gemini or OpenAI based on request.model.
    """
    # For debugging and user identification later:
    # print(f"Incoming request headers: {fastapi_request.headers}")
    # print(f"Chat completion request for model: {request.model}, User: {request.user}")
    # print(f"Request Body: {request.model_dump_json(indent=2)}")

    request_id = f"chatcmpl-{uuid.uuid4()}" # For streaming and non-streaming responses

    # --- Route to Ollama ---
    if request.model.startswith("ollama/"):
        if request.stream:
            return StreamingResponse(
                ollama_chat_stream_generator(request.model, request.messages, request_id),
                media_type="text/event-stream"
            )
        else: # Non-streaming for Ollama
            ollama_model_name = request.model.replace("ollama/", "")
            ollama_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            payload = {"model": ollama_model_name, "messages": ollama_messages, "stream": False}
            try:
                response = await http_client.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=120.0)
                response.raise_for_status()
                ollama_data = response.json()
                
                final_content = ollama_data.get("message", {}).get("content", "")
                # These are approximations from Ollama's /api/chat non-streamed response
                prompt_tokens = ollama_data.get("prompt_eval_count", 0) 
                completion_tokens = ollama_data.get("eval_count", 0)
                total_tokens = prompt_tokens + completion_tokens
                
                return ChatCompletionResponse(
                    id=request_id,
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=final_content), finish_reason="stop")],
                    usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens) if total_tokens > 0 else None
                )
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=f"Ollama API Error (non-streaming): {e.response.text}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error with Ollama non-streaming: {str(e)}")
    
    # --- Placeholder for Gemini ---
    elif request.model.startswith("gemini/"):
        # TODO: Implement Gemini call (streaming and non-streaming)
        # Remember to handle API key from settings.GEMINI_API_KEY
        print(f"Gemini model requested: {request.model}. User from request: {request.user}") # Log user for Gemini
        raise HTTPException(status_code=501, detail=f"Gemini model '{request.model}' handling not yet implemented.")

    # --- Placeholder for OpenAI ---
    elif request.model.startswith("openai/"):
        # TODO: Implement OpenAI call (streaming and non-streaming)
        # Remember to handle API key from settings.OPENAI_API_KEY
        print(f"OpenAI model requested: {request.model}. User from request: {request.user}") # Log user for OpenAI
        raise HTTPException(status_code=501, detail=f"OpenAI model '{request.model}' handling not yet implemented.")
        
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")


# Keep your existing /test_ollama_prompt if you find it useful for direct Ollama testing
class OllamaSimplePromptRequest(BaseModel): # Make sure this is defined or imported
    model: str = "gemma:2b"
    prompt: str
    stream: bool = False

@app.post("/test_ollama_prompt", tags=["Ollama Tests"])
async def test_ollama_prompt(request_data: OllamaSimplePromptRequest):
    # (Your existing code for /test_ollama_prompt from Step #2)
    # ... (ensure it uses settings.OLLAMA_BASE_URL and http_client) ...
    # For brevity, I'm not pasting it again here, but ensure it's functional
    # or adapt it to use the /api/chat structure like the main completions endpoint.
    # Simplified version:
    ollama_generate_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {"model": request_data.model, "prompt": request_data.prompt, "stream": request_data.stream}
    try:
        response = await http_client.post(ollama_generate_url, json=payload, timeout=120.0)
        response.raise_for_status()
        ollama_response = response.json()
        return {"model_used": request_data.model, "ollama_response": ollama_response.get("response", ""),"full_ollama_payload": ollama_response}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Ollama API Error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Request error connecting to Ollama: {str(e)}")