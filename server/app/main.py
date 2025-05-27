# server/app/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, Depends # Added Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session # Added Session for type hinting
import httpx
import json
import time
import uuid
from typing import List, Dict, Optional, Union # Ensure List is here

from .config import settings
from .openai_models import (
    ModelCard, ModelList, ChatCompletionRequest, ChatMessageInput,
    ChatCompletionStreamResponse, ChatCompletionStreamChoice, ChatCompletionStreamChoiceDelta,
    ChatCompletionResponse, ChatCompletionChoice, ChatMessageOutput, Usage
)
# Database and Service imports
from .db.database import create_db_and_tables, get_db, SessionLocal # Import SessionLocal for direct use in startup
from .db import models
from .services import chat_service # Import your new service

app = FastAPI(title="RAG Workspace Server")
http_client = httpx.AsyncClient()

@app.on_event("startup")
async def startup_event():
    global http_client
    if http_client.is_closed:
        http_client = httpx.AsyncClient()
    
    print("INFO:     Starting up RAG Server...")
    # Now that 'app.db.models' has been imported at the module level of main.py,
    # Base.metadata within create_db_and_tables() will know about all your tables.
    create_db_and_tables() 
    
    db: Session = SessionLocal()
    try:
        project_name = f"{settings.WORKSPACE_NAME} Project"
        project = chat_service.get_or_create_project(db, project_name=project_name, description=f"Default project for workspace '{settings.WORKSPACE_NAME}'")
        
        admin_user_identifier = settings.WEBUI_ADMIN_USER_EMAIL
        if not admin_user_identifier: 
            admin_user_identifier = f"admin_{settings.WORKSPACE_NAME}"
            print(f"WARNING:  WEBUI_ADMIN_USER_EMAIL not set in .env, using '{admin_user_identifier}' as default admin for project linking.")

        user = chat_service.get_or_create_user(db, user_identifier=admin_user_identifier)
        chat_service.ensure_user_linked_to_project(db, user_id=user.id, project_id=project.id)
        print(f"INFO:     Default project '{project.name}' (ID: {project.id}) checked/created and linked to user '{user.id}'.")
        
    except Exception as e:
        print(f"ERROR:    Error during startup project initialization: {e}")
        # Print the full traceback for the exception during startup
        import traceback
        traceback.print_exc()
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
    # (Keep this function as it was in Step #3, listing your models)
    # For example:
    available_models = [
        ModelCard(id="ollama/gemma:2b", owned_by="Gen AI Enable"),
        ModelCard(id="ollama/qwen:1.8b", owned_by="Gen AI Enable"), # Using your specified tags
        ModelCard(id="ollama/llama3:70b", owned_by="Gen AI Enable"),
    ]
    if settings.GEMINI_API_KEY:
        available_models.append(ModelCard(id="gemini/gemini-1.5-flash-latest", owned_by="Gen AI Enable"))
    if settings.OPENAI_API_KEY:
        available_models.append(ModelCard(id="openai/gpt-3.5-turbo", owned_by="Gen AI Enable"))
    return ModelList(data=available_models)


async def ollama_chat_stream_generator(
    db: Session, # DB session passed in
    project_id_for_chat: str, # Project ID for context
    user_identifier: str, # From OWI, e.g., {{USER_NAME}}
    conversation_id_from_owi: str, # OWI's chat_id, used as our Conversation.id
    model_id: str, 
    messages_for_llm: List[ChatMessageInput], # Full context from OWI request
    request_id: str
):
    # Save the current user message (assumed to be the last one in messages_for_llm)
    if messages_for_llm and messages_for_llm[-1].role == "user":
        user_message_content = messages_for_llm[-1].content
        chat_service.add_message_to_database(
            db=db,
            conversation_id=conversation_id_from_owi,
            author_user_id=user_identifier,
            role="user",
            content=user_message_content
        )

    ollama_model_name = model_id.replace("ollama/", "")
    ollama_formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm]
    payload = {"model": ollama_model_name, "messages": ollama_formatted_messages, "stream": True}
    
    assistant_response_buffer = ""
    try:
        async with http_client.stream("POST", f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=120.0) as response:
            if response.status_code != 200:
                error_content = await response.aread()
                error_msg = f"Ollama API Error ({response.status_code}): {error_content.decode()}"
                print(f"ERROR:    Stream - {error_msg}")
                error_chunk_data = ChatCompletionStreamResponse(id=request_id,model=model_id,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f"Error from Ollama: Status {response.status_code}"),finish_reason="error")])
                yield f"data: {error_chunk_data.model_dump_json()}\n\n"
                # No return here, fall through to finally block to send [DONE]
            else: # Process stream only if status is 200
                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            delta_content = chunk.get("message", {}).get("content", "")
                            
                            if delta_content is not None:
                                 assistant_response_buffer += delta_content
                            
                            delta_role_to_send = None
                            # Send role="assistant" only for the first non-empty content chunk from assistant
                            if chunk.get("message",{}).get("role") == "assistant" and delta_content:
                                if len(assistant_response_buffer.strip()) == len(delta_content.strip()): # True if this is the first content
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
        
        if assistant_response_buffer and (response is None or response.status_code == 200): # Only save if successful stream
            chat_service.add_message_to_database(
                db=db, conversation_id=conversation_id_from_owi,
                author_user_id=None, role="assistant", content=assistant_response_buffer, model_used=model_id
            )
    
    except httpx.RequestError as e:
        error_msg = f"Request error connecting to Ollama for streaming: {str(e)}"
        print(f"ERROR:    Stream - {error_msg}")
        error_chunk_data = ChatCompletionStreamResponse(id=request_id,model=model_id,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f"Error: {error_msg}"),finish_reason="error")])
        yield f"data: {error_chunk_data.model_dump_json()}\n\n"
    except Exception as e:
        error_msg = f"Generic error in Ollama stream generator: {str(e)}"
        print(f"ERROR:    Stream - {error_msg}")
        error_chunk_data = ChatCompletionStreamResponse(id=request_id,model=model_id,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f"Error: {error_msg}"),finish_reason="error")])
        yield f"data: {error_chunk_data.model_dump_json()}\n\n"
    finally:
        yield f"data: [DONE]\n\n"


@app.post("/v1/chat/completions", tags=["OpenAI Compatibility"])
async def chat_completions(
    request: ChatCompletionRequest, 
    fastapi_request: FastAPIRequest, # To inspect raw request for custom OWI fields
    db: Session = Depends(get_db)   # Inject DB session
):
    raw_request_body = await fastapi_request.json()
    
    # --- User Identification ---
    user_identifier = "guest_user" # Default
    owi_variables = raw_request_body.get("variables", {}) 
    if owi_variables and isinstance(owi_variables, dict) and owi_variables.get("{{USER_NAME}}"): # Check type of owi_variables
        user_identifier = owi_variables["{{USER_NAME}}"]
    elif request.user: # Standard OpenAI 'user' field as fallback
        user_identifier = request.user
    
    db_user = chat_service.get_or_create_user(db, user_identifier=user_identifier)

    # --- Project Context ---
    # For v1, all chats go to the default project created on startup.
    project_name_for_chat = f"{settings.WORKSPACE_NAME} Project" # As created in startup_event
    project = chat_service.get_or_create_project(db, project_name=project_name_for_chat) # Ensures it exists
    chat_service.ensure_user_linked_to_project(db, user_id=db_user.id, project_id=project.id)

    # --- Conversation ID & Management ---
    # OpenWebUI sends a unique `chat_id` in its custom payload (top level of raw_request_body).
    conversation_id_from_owi = raw_request_body.get("chat_id")
    if not conversation_id_from_owi:
        # For a truly new chat, OWI might not send a chat_id on the very first message.
        # The server should generate one and the UI should ideally pick it up from the response for subsequent messages.
        # For now, if OWI doesn't send one, we generate one for backend persistence.
        conversation_id_from_owi = str(uuid.uuid4())
        print(f"INFO:     'chat_id' not found in OWI request for user '{db_user.id}'. Generated new backend conversation ID: {conversation_id_from_owi}")
    
    # Ensure the conversation record exists in our DB, linked to the correct project and user
    chat_service.get_or_create_conversation(
        db, 
        conversation_id_from_owi=conversation_id_from_owi, 
        project_id=project.id, 
        creator_user_id=db_user.id,
        # OWI might send the title of an existing chat if it's resuming.
        # Or it might send the first user message as the title for a new chat.
        # Using `raw_request_body.get("title")` or deriving from first message could be options.
        title=raw_request_body.get("title", "New Chat from " + db_user.id) 
    )
    
    # --- Prepare messages for LLM ---
    # OpenWebUI sends the list of messages it wants the LLM to process in `request.messages`.
    # This list already includes relevant history + the latest user message.
    messages_for_llm = [ChatMessageInput(role=m.role, content=m.content) for m in request.messages]
    
    request_id = f"chatcmpl-{uuid.uuid4()}" # For OpenAI compatibility

    # --- Route to LLM ---
    if request.model.startswith("ollama/"):
        if request.stream:
            return StreamingResponse(
                ollama_chat_stream_generator(
                    db, project.id, db_user.id, conversation_id_from_owi,
                    request.model, messages_for_llm, request_id
                ), 
                media_type="text/event-stream"
            )
        else: # Non-streaming for Ollama
            # Save current user message (last one in messages_for_llm)
            if messages_for_llm and messages_for_llm[-1].role == "user":
                user_msg_content = messages_for_llm[-1].content
                chat_service.add_message_to_database(db, conversation_id_from_owi, "user", user_msg_content, author_user_id=db_user.id)

            ollama_model_name = request.model.replace("ollama/", "")
            ollama_formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm]
            payload = {"model": ollama_model_name, "messages": ollama_formatted_messages, "stream": False}
            
            try:
                response = await http_client.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=120.0)
                response.raise_for_status()
                ollama_data = response.json()
                
                final_content = ollama_data.get("message", {}).get("content", "")
                if final_content: # Save assistant response
                    chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", final_content, author_user_id=None, model_used=request.model)
                
                prompt_tokens = ollama_data.get("prompt_eval_count", 0) 
                completion_tokens = ollama_data.get("eval_count", 0)
                total_tokens = prompt_tokens + completion_tokens
                
                return ChatCompletionResponse(
                    id=request_id, model=request.model,
                    choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=final_content), finish_reason="stop")],
                    usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens) if total_tokens > 0 else None
                )
            except httpx.HTTPStatusError as e:
                print(f"ERROR:    Ollama API Error (non-streaming): {e.response.status_code} - {e.response.text}")
                raise HTTPException(status_code=e.response.status_code, detail=f"Ollama API Error: {e.response.text}")
            except Exception as e:
                print(f"ERROR:    Error with Ollama non-streaming: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing Ollama non-streaming request: {str(e)}")
    
    # --- Placeholder for Gemini/OpenAI (ensure they also save to DB) ---
    elif request.model.startswith("gemini/") or request.model.startswith("openai/"):
        # Save user message
        if messages_for_llm and messages_for_llm[-1].role == "user":
             user_msg_content = messages_for_llm[-1].content
             chat_service.add_message_to_database(db, conversation_id_from_owi, "user", user_msg_content, author_user_id=db_user.id)
        
        mock_response_content = f"LLM service for {request.model} not fully implemented. Message received and conceptually saved."
        # Save mock assistant response
        chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", mock_response_content, author_user_id=None, model_used=request.model)

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
                usage=Usage(prompt_tokens=10, completion_tokens=len(mock_response_content)//4 if mock_response_content else 0, total_tokens=10 + (len(mock_response_content)//4 if mock_response_content else 0)) # Dummy usage
            )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")

# Keep or remove /test_ollama_prompt as needed. If kept, ensure OllamaSimplePromptRequest Pydantic model is defined.
# class OllamaSimplePromptRequest(BaseModel): model: str = "gemma:2b"; prompt: str; stream: bool = False
# @app.post("/test_ollama_prompt", tags=["Ollama Tests"]) ...