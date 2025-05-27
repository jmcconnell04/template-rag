# server/app/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel # <--- ADD THIS LINE or ensure BaseModel is in your pydantic imports
import httpx
import json
import time
import uuid
from typing import List, Dict, Optional, Union

# ... rest of your imports ...
from .config import settings
from .openai_models import (
    ModelCard, ModelList, ChatCompletionRequest, ChatMessageInput,
    ChatCompletionStreamResponse, ChatCompletionStreamChoice, ChatCompletionStreamChoiceDelta,
    ChatCompletionResponse, ChatCompletionChoice, ChatMessageOutput, Usage
)
from .db.database import create_db_and_tables, get_db, SessionLocal
from .services import chat_service

app = FastAPI(title="RAG Workspace Server")
http_client = httpx.AsyncClient()

@app.on_event("startup")
async def startup_event():
    global http_client
    if http_client.is_closed:
        http_client = httpx.AsyncClient()
    
    print("INFO:     Starting up RAG Server...")
    create_db_and_tables()
    
    db: Session = SessionLocal()
    try:
        project_name = f"{settings.WORKSPACE_NAME} Project"
        project = chat_service.get_or_create_project(db, project_name=project_name, description=f"Default project for workspace '{settings.WORKSPACE_NAME}'")
        
        default_user_for_project = settings.WEBUI_ADMIN_USER_EMAIL if hasattr(settings, 'WEBUI_ADMIN_USER_EMAIL') and settings.WEBUI_ADMIN_USER_EMAIL else f"admin_{settings.WORKSPACE_NAME}"
        if not default_user_for_project: # Fallback if WEBUI_ADMIN_USER_EMAIL is empty
            default_user_for_project = f"admin_{settings.WORKSPACE_NAME}"
            
        user = chat_service.get_or_create_user(db, username=default_user_for_project)
        chat_service.ensure_user_linked_to_project(db, user_id=user.id, project_id=project.id)
        print(f"INFO:     Default project '{project.name}' (ID: {project.id}) checked/created and linked to user '{user.id}'.")
        
    except Exception as e:
        print(f"ERROR:    Error during startup project initialization: {e}")
    finally:
        db.close()
        
    print(f"INFO:     RAG Server for '{settings.WORKSPACE_NAME}' started. Ollama URL: '{settings.OLLAMA_BASE_URL}'. DB Initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
    print(f"INFO:     RAG Server for {settings.WORKSPACE_NAME} shutting down.")

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok", "message": f"RAG Server for {settings.WORKSPACE_NAME} is healthy"}

@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": f"Welcome to {settings.WORKSPACE_NAME} RAG Server!",
        "ollama_base_url": settings.OLLAMA_BASE_URL,
    }

@app.get("/v1/models", response_model=ModelList, tags=["OpenAI Compatibility"])
async def list_models():
    available_models = [
        ModelCard(id="ollama/gemma:2b", owned_by="Gen AI Enable"),
        ModelCard(id="ollama/qwen:1.8b", owned_by="Gen AI Enable"),
        ModelCard(id="ollama/llama3:70b", owned_by="Gen AI Enable"),
    ]
    if settings.GEMINI_API_KEY:
        available_models.append(ModelCard(id="gemini/gemini-1.5-flash-latest", owned_by="Gen AI Enable"))
    if settings.OPENAI_API_KEY:
        available_models.append(ModelCard(id="openai/gpt-3.5-turbo", owned_by="Gen AI Enable"))
    return ModelList(data=available_models)

async def ollama_chat_stream_generator(
    db: Session,
    project_id_for_chat: str,
    user_identifier: str,
    conversation_id_from_owi: str,
    model_id: str, 
    messages_for_llm: List[ChatMessageInput],
    request_id: str
):
    if messages_for_llm and messages_for_llm[-1].role == "user":
        user_message_content = messages_for_llm[-1].content
        chat_service.add_message_to_database( # Corrected function name
            db=db, conversation_id=conversation_id_from_owi,
            author_user_id=user_identifier, role="user", content=user_message_content
        )

    ollama_model_name = model_id.replace("ollama/", "")
    ollama_formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm]
    payload = {"model": ollama_model_name, "messages": ollama_formatted_messages, "stream": True}
    
    assistant_response_buffer = ""
    try: # Outer try for the entire streaming operation
        async with http_client.stream("POST", f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=120.0) as response:
            if response.status_code != 200:
                error_content = await response.aread()
                error_msg = f"Ollama API Error ({response.status_code}): {error_content.decode()}"
                print(f"ERROR:    {error_msg}")
                error_chunk_data = ChatCompletionStreamResponse(id=request_id,model=model_id,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f"Error from Ollama: Status {response.status_code}"),finish_reason="error")])
                yield f"data: {error_chunk_data.model_dump_json()}\n\n"
                # No return here, fall through to finally send [DONE]

            else: # Only process stream if status is 200
                async for line in response.aiter_lines():
                    if line:
                        try: # Inner try for JSON decoding each line
                            chunk = json.loads(line)
                            delta_content = chunk.get("message", {}).get("content", "")
                            
                            if delta_content is not None:
                                 assistant_response_buffer += delta_content
                            
                            delta_role_to_send = None
                            if chunk.get("message",{}).get("role") == "assistant" and delta_content:
                                # Send role only for the first non-empty delta from assistant
                                if len(assistant_response_buffer) == len(delta_content if delta_content else ""):
                                     delta_role_to_send = "assistant"

                            stream_choice = ChatCompletionStreamChoice(
                                index=0,
                                delta=ChatCompletionStreamChoiceDelta(content=delta_content if delta_content else None, role=delta_role_to_send),
                                finish_reason="stop" if chunk.get("done") else None
                            )
                            stream_resp = ChatCompletionStreamResponse(id=request_id, model=model_id, choices=[stream_choice])
                            yield f"data: {stream_resp.model_dump_json()}\n\n"
                            
                            if chunk.get("done"):
                                break 
                        except json.JSONDecodeError:
                            print(f"WARNING:  Could not decode JSON line from Ollama stream: {line}")
                            continue
        
        # This part is after 'async with', but still within the main 'try' block
        if assistant_response_buffer and response.status_code == 200 : # Only save if successful stream
            chat_service.add_message_to_database( # Corrected function name
                db=db, conversation_id=conversation_id_from_owi,
                author_user_id=None, role="assistant", content=assistant_response_buffer, model_used=model_id
            )
    
    except httpx.RequestError as e: # Handles network errors, timeouts to Ollama
        error_msg = f"Request error connecting to Ollama: {str(e)}"
        print(f"ERROR:    {error_msg}")
        error_chunk_data = ChatCompletionStreamResponse(id=request_id,model=model_id,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f"Error: {error_msg}"),finish_reason="error")])
        yield f"data: {error_chunk_data.model_dump_json()}\n\n"
    except Exception as e: # Handles any other unexpected errors during streaming logic
        error_msg = f"Generic error in Ollama stream generator: {str(e)}"
        print(f"ERROR:    {error_msg}")
        error_chunk_data = ChatCompletionStreamResponse(id=request_id,model=model_id,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f"Error: {error_msg}"),finish_reason="error")])
        yield f"data: {error_chunk_data.model_dump_json()}\n\n"
    finally: # Ensure [DONE] is always sent to close the client-side event stream properly
        yield f"data: [DONE]\n\n"


@app.post("/v1/chat/completions", tags=["OpenAI Compatibility"])
async def chat_completions(
    request: ChatCompletionRequest, 
    fastapi_request: FastAPIRequest,
    db: Session = Depends(get_db)
):
    raw_request_body = await fastapi_request.json()
    
    user_identifier = "guest_user"
    owi_variables = raw_request_body.get("variables", {})
    if owi_variables and isinstance(owi_variables, dict) and owi_variables.get("{{USER_NAME}}"):
        user_identifier = owi_variables["{{USER_NAME}}"]
    elif request.user:
        user_identifier = request.user
    
    db_user = chat_service.get_or_create_user(db, username=user_identifier)

    project_name_for_chat = f"{settings.WORKSPACE_NAME} Project"
    project = chat_service.get_or_create_project(db, project_name=project_name_for_chat)
    chat_service.ensure_user_linked_to_project(db, user_id=db_user.id, project_id=project.id)

    conversation_id_from_owi = raw_request_body.get("chat_id")
    if not conversation_id_from_owi:
        conversation_id_from_owi = str(uuid.uuid4())
        print(f"INFO:     'chat_id' not found in OWI request. Generated new backend ID: {conversation_id_from_owi} for user '{db_user.id}' in project '{project.id}'")
    
    chat_service.get_or_create_conversation(
        db, 
        conversation_id_from_owi=conversation_id_from_owi, 
        project_id=project.id, 
        creator_user_id=db_user.id,
        title=raw_request_body.get("title", "New Chat")
    )
    
    messages_for_llm = [ChatMessageInput(role=m.role, content=m.content) for m in request.messages]
    request_id = f"chatcmpl-{uuid.uuid4()}"

    if request.model.startswith("ollama/"):
        if request.stream:
            return StreamingResponse(
                ollama_chat_stream_generator(
                    db, project.id, db_user.id, conversation_id_from_owi,
                    request.model, messages_for_llm, request_id
                ), 
                media_type="text/event-stream"
            )
        else: 
            if messages_for_llm and messages_for_llm[-1].role == "user":
                user_msg_content = messages_for_llm[-1].content
                chat_service.add_message_to_database(db, conversation_id_from_owi, "user", user_msg_content, author_user_id=db_user.id) # Corrected

            ollama_model_name = request.model.replace("ollama/", "")
            ollama_formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm]
            payload = {"model": ollama_model_name, "messages": ollama_formatted_messages, "stream": False}
            
            try: # Try for non-streaming Ollama call
                response = await http_client.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=120.0)
                response.raise_for_status()
                ollama_data = response.json()
                
                final_content = ollama_data.get("message", {}).get("content", "")
                if final_content: 
                    chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", final_content, model_used=request.model, author_user_id=None) # Corrected
                
                prompt_tokens = ollama_data.get("prompt_eval_count", 0) 
                completion_tokens = ollama_data.get("eval_count", 0)
                total_tokens = prompt_tokens + completion_tokens
                
                return ChatCompletionResponse(
                    id=request_id, model=request.model,
                    choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=final_content), finish_reason="stop")],
                    usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens) if total_tokens > 0 else None
                )
            except httpx.HTTPStatusError as e: # Corresponding except
                print(f"ERROR: Ollama API Error (non-streaming): {e.response.status_code} - {e.response.text}")
                raise HTTPException(status_code=e.response.status_code, detail=f"Ollama API Error: {e.response.text}")
            except Exception as e: # Corresponding except
                print(f"ERROR: Error with Ollama non-streaming: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing Ollama non-streaming request: {str(e)}")
    
    elif request.model.startswith("gemini/") or request.model.startswith("openai/"):
        if messages_for_llm and messages_for_llm[-1].role == "user":
             user_msg_content = messages_for_llm[-1].content
             chat_service.add_message_to_database(db, conversation_id_from_owi, "user", user_msg_content, author_user_id=db_user.id) # Corrected
        
        mock_response_content = f"LLM service for {request.model} not fully implemented. Message received and conceptually saved."
        chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", mock_response_content, model_used=request.model, author_user_id=None) # Corrected

        if request.stream: 
            async def mock_stream_placeholder():
                chunk_data = ChatCompletionStreamResponse(id=request_id, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(content=mock_response_content, role='assistant'))])
                yield f"data: {chunk_data.model_dump_json()}\n\n"
                finish_data = ChatCompletionStreamResponse(id=request_id, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(), finish_reason='stop')])
                yield f"data: {finish_data.model_dump_json()}\n\n"
                yield f"data: [DONE]\n\n"
            return StreamingResponse(mock_stream_placeholder(), media_type="text/event-stream")
        else: 
             return ChatCompletionResponse(
                id=request_id, model=request.model,
                choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=mock_response_content), finish_reason="stop")],
                usage=Usage(prompt_tokens=10, completion_tokens=len(mock_response_content)//4 if mock_response_content else 0, total_tokens=10 + (len(mock_response_content)//4 if mock_response_content else 0))
            )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")

# Ensure OllamaSimplePromptRequest is defined if /test_ollama_prompt is kept for direct testing
# (This model was defined in the previous iteration's main.py, ensure it's still there or add it if needed)
class OllamaSimplePromptRequest(BaseModel): # This line uses BaseModel
    model: str = "gemma:2b"
    prompt: str
    stream: bool = False

@app.post("/test_ollama_prompt", tags=["Ollama Tests"])
async def test_ollama_prompt(request_data: OllamaSimplePromptRequest):
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