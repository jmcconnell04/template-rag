# server/app/main.py
from .services import chat_service, rag_service # Add rag_service
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

app = FastAPI(title="RAG Workspace Server")
http_client = httpx.AsyncClient()

@app.on_event("startup")
async def startup_event():
    # ... (existing http_client init, create_db_and_tables) ...
    print("INFO:     RAG Server on_event[startup] triggered.")
    create_db_and_tables()
    
    db: Session = SessionLocal()
    try:
        project_name = f"{settings.WORKSPACE_NAME} Project"
        project = chat_service.get_or_create_project(db, project_name=project_name, description=f"Default project for workspace '{settings.WORKSPACE_NAME}'")
        
        admin_user_identifier = settings.WEBUI_ADMIN_USER_EMAIL
        if not admin_user_identifier:
            admin_user_identifier = f"admin_{settings.WORKSPACE_NAME}"
            print(f"WARNING:  WEBUI_ADMIN_USER_EMAIL not set, using '{admin_user_identifier}' for project linking.")
        user = chat_service.get_or_create_user(db, user_identifier=admin_user_identifier)
        chat_service.ensure_user_linked_to_project(db, user_id=user.id, project_id=project.id)
        print(f"INFO:     Default project '{project.name}' (ID: {project.id}) checked/created and linked to user '{user.id}'.")

        # Auto-seeding logic for RAG
        if settings.AUTO_SEED_ON_STARTUP:
            print(f"INFO:     AUTO_SEED_ON_STARTUP is true. Attempting to seed documents for project '{project.id}'...")
            seed_directory_in_container = "/app/seed_data" # This path is mounted from ./rag_files/seed/
            rag_service.seed_documents_from_directory(db, project_id=project.id, seed_dir_path=seed_directory_in_container)
        else:
            print("INFO:     AUTO_SEED_ON_STARTUP is false. Skipping RAG seed data ingestion.")
            
    except Exception as e:
        print(f"ERROR:    Error during startup RAG/project initialization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
    print(f"INFO:     RAG Server for '{settings.WORKSPACE_NAME}' started. Ollama: '{settings.OLLAMA_BASE_URL}'. DB & RAG Service Initialized.")



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
    db: Session, project_id_for_chat: str, user_identifier: str,
    conversation_id_from_owi: str, model_id: str, 
    messages_for_llm_with_rag_context: List[ChatMessageInput], # MODIFIED: Use this augmented list
    request_id: str
):
    # Save the *original* user message (last message from the *original* request before RAG augmentation)
    # This assumes messages_for_llm_with_rag_context still contains the original user message as its last user turn.
    # A cleaner way might be to pass original_user_message separately if augmentation modifies it heavily.
    # For now, let's assume the last user message in messages_for_llm_with_rag_context is what needs to be saved.
    # This part needs careful thought: what exactly are we saving as the "user message" if the prompt to LLM is augmented?
    # Let's save the original user message from the *actual request* before augmentation.
    # This will be handled in the main chat_completions endpoint before this generator is called.

    ollama_model_name = model_id.replace("ollama/", "")
    # Convert augmented messages to Ollama format
    ollama_formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm_with_rag_context]
    payload = {"model": ollama_model_name, "messages": ollama_formatted_messages, "stream": True}
    
    assistant_response_buffer = ""
    try:
        async with http_client.stream("POST", f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=120.0) as response:
            # ... (rest of the streaming logic and error handling from Step #4, now using ollama_formatted_messages) ...
            # Important: When saving assistant response, use the original conversation_id_from_owi
            if response.status_code != 200:
                error_content = await response.aread()
                error_msg = f"Ollama API Error ({response.status_code}): {error_content.decode()}"
                print(f"ERROR:    Stream - {error_msg}")
                error_chunk_data = ChatCompletionStreamResponse(id=request_id,model=model_id,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f"Error from Ollama: Status {response.status_code}"),finish_reason="error")])
                yield f"data: {error_chunk_data.model_dump_json()}\n\n"
            else:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            delta_content = chunk.get("message", {}).get("content", "")
                            if delta_content is not None: assistant_response_buffer += delta_content
                            delta_role_to_send = "assistant" if chunk.get("message",{}).get("role") == "assistant" and delta_content and len(assistant_response_buffer.strip()) == len(delta_content.strip()) else None
                            stream_choice = ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=delta_content if delta_content else None, role=delta_role_to_send),finish_reason="stop" if chunk.get("done") else None)
                            stream_resp = ChatCompletionStreamResponse(id=request_id, model=model_id, choices=[stream_choice])
                            yield f"data: {stream_resp.model_dump_json()}\n\n"
                            if chunk.get("done"): break
                        except json.JSONDecodeError:
                            print(f"WARNING:  Could not decode JSON line from Ollama stream: {line}")
                            continue
        
        if assistant_response_buffer and (getattr(response, 'status_code', 500) == 200):
            chat_service.add_message_to_database(
                db=db, conversation_id=conversation_id_from_owi,
                author_user_id=None, role="assistant", content=assistant_response_buffer, model_used=model_id
            )
    # ... (Outer try-except-finally blocks from Step #4 to yield [DONE]) ...
    except httpx.RequestError as e: # Simplified error handling for brevity
        error_msg = f"Request error (stream): {str(e)}"
        print(f"ERROR: {error_msg}")
        error_chunk_data = ChatCompletionStreamResponse(id=request_id,model=model_id,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f"Error: {error_msg}"),finish_reason="error")])
        yield f"data: {error_chunk_data.model_dump_json()}\n\n"
    except Exception as e:
        error_msg = f"Generic error (stream): {str(e)}"
        print(f"ERROR: {error_msg}")
        error_chunk_data = ChatCompletionStreamResponse(id=request_id,model=model_id,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f"Error: {error_msg}"),finish_reason="error")])
        yield f"data: {error_chunk_data.model_dump_json()}\n\n"
    finally:
        yield f"data: [DONE]\n\n"


@app.post("/v1/chat/completions", tags=["OpenAI Compatibility"])
async def chat_completions(
    request: ChatCompletionRequest, 
    fastapi_request: FastAPIRequest,
    db: Session = Depends(get_db)
):
    raw_request_body = await fastapi_request.json()
    user_identifier, project, conversation_id_from_owi, db_user = await _get_request_context(raw_request_body, request, db) # Helper function

    # Extract the original user messages from the incoming request
    original_user_messages = [ChatMessageInput(role=m.role, content=m.content) for m in request.messages]

    # --- Save current user message (last one in the original list) to DB ---
    if original_user_messages and original_user_messages[-1].role == "user":
        user_msg_content = original_user_messages[-1].content
        chat_service.add_message_to_database(
            db, 
            conversation_id=conversation_id_from_owi, 
            author_user_id=db_user.id, 
            role="user", 
            content=user_msg_content
        )
    
    # --- RAG Enhancement ---
    messages_for_llm_with_rag_context = list(original_user_messages) # Start with original messages
    if original_user_messages and original_user_messages[-1].role == "user":
        current_user_query = original_user_messages[-1].content
        try:
            relevant_chunks = rag_service.query_project_collection(
                project_id=project.id, 
                query_text=current_user_query,
                n_results=settings.RAG_TOP_K
            )
            if relevant_chunks:
                retrieved_context_str = "\n\n--- Relevant Context Retrieved ---\n" + "\n\n".join(relevant_chunks) + "\n--- End of Context ---\n"
                print(f"INFO:     RAG: Retrieved {len(relevant_chunks)} chunks for query in project '{project.id}'.")
                # Augment the prompt for the LLM
                # Option: Create a new system message with context and prepend it
                messages_for_llm_with_rag_context.insert(
                    len(messages_for_llm_with_rag_context) - 1, # Insert before the last user message
                    ChatMessageInput(role="system", content=f"Based on the following information, answer the user's query:\n{retrieved_context_str}")
                )
                # Or augment the last user message:
                # messages_for_llm_with_rag_context[-1].content = f"{retrieved_context_str}\n\nUser Query: {current_user_query}"
            else:
                print(f"INFO:     RAG: No relevant chunks found for project '{project.id}'.")
        except Exception as e:
            print(f"ERROR:    RAG: Failed to retrieve context for project '{project.id}': {e}")
            # Proceed without RAG context if it fails

    request_id = f"chatcmpl-{uuid.uuid4()}"

    # --- Route to LLM ---
    if request.model.startswith("ollama/"):
        if request.stream:
            return StreamingResponse(
                ollama_chat_stream_generator(
                    db, project.id, db_user.id, conversation_id_from_owi,
                    request.model, messages_for_llm_with_rag_context, request_id # Pass augmented messages
                ), 
                media_type="text/event-stream"
            )
        else: # Non-streaming for Ollama
            ollama_model_name = request.model.replace("ollama/", "")
            ollama_formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm_with_rag_context]
            payload = {"model": ollama_model_name, "messages": ollama_formatted_messages, "stream": False}
            try:
                response = await http_client.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=120.0)
                response.raise_for_status()
                ollama_data = response.json()
                final_content = ollama_data.get("message", {}).get("content", "")
                if final_content:
                    chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", final_content, author_user_id=None, model_used=request.model)
                # ... (construct and return ChatCompletionResponse as before) ...
                prompt_tokens = ollama_data.get("prompt_eval_count", 0); completion_tokens = ollama_data.get("eval_count", 0)
                total_tokens = prompt_tokens + completion_tokens
                return ChatCompletionResponse(id=request_id, model=request.model, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=final_content), finish_reason="stop")], usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens) if total_tokens > 0 else None)
            except httpx.HTTPStatusError as e: # ... (error handling) ...
                print(f"ERROR:    Ollama API Error (non-streaming): {e.response.status_code} - {e.response.text}")
                raise HTTPException(status_code=e.response.status_code, detail=f"Ollama API Error: {e.response.text}")
            except Exception as e: # ... (error handling) ...
                print(f"ERROR:    Error with Ollama non-streaming: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing Ollama non-streaming request: {str(e)}")

    # ... (Gemini/OpenAI placeholders - they should also use messages_for_llm_with_rag_context and save assistant messages) ...
    elif request.model.startswith("gemini/") or request.model.startswith("openai/"):
        # User message was already saved before RAG step
        mock_response_content = f"LLM service for {request.model} (with RAG context if any) not fully implemented. Message received."
        chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", mock_response_content, author_user_id=None, model_used=request.model)
        # ... (return mock stream/non-stream as before)
        if request.stream: 
            async def mock_stream_placeholder(): # Simplified mock stream
                chunk_data = ChatCompletionStreamResponse(id=request_id, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(content=mock_response_content, role='assistant'))])
                yield f"data: {chunk_data.model_dump_json()}\n\n"
                finish_data = ChatCompletionStreamResponse(id=request_id, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(), finish_reason='stop')])
                yield f"data: {finish_data.model_dump_json()}\n\n"
                yield f"data: [DONE]\n\n"
            return StreamingResponse(mock_stream_placeholder(), media_type="text/event-stream")
        else: # Mock non-stream
             return ChatCompletionResponse(
                id=request_id, model=request.model,
                choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=mock_response_content), finish_reason="stop")],
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20) 
            )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")

async def _get_request_context(raw_request_body: dict, request: ChatCompletionRequest, db: Session):
    """Helper to extract user, project, and conversation context."""
    user_identifier = "guest_user"
    owi_variables = raw_request_body.get("variables", {})
    if owi_variables and isinstance(owi_variables, dict) and owi_variables.get("{{USER_NAME}}"):
        user_identifier = owi_variables["{{USER_NAME}}"]
    elif request.user:
        user_identifier = request.user
    
    db_user = chat_service.get_or_create_user(db, user_identifier=user_identifier)
    
    project_name_for_chat = f"{settings.WORKSPACE_NAME} Project"
    project = chat_service.get_or_create_project(db, project_name=project_name_for_chat)
    chat_service.ensure_user_linked_to_project(db, user_id=db_user.id, project_id=project.id)

    conversation_id_from_owi = raw_request_body.get("chat_id")
    if not conversation_id_from_owi:
        conversation_id_from_owi = str(uuid.uuid4())
        print(f"INFO:     'chat_id' not found in OWI request for user '{db_user.id}'. Generated new backend conversation ID: {conversation_id_from_owi}")
    
    chat_service.get_or_create_conversation(
        db, 
        conversation_id_from_owi=conversation_id_from_owi, 
        project_id=project.id, 
        creator_user_id=db_user.id,
        title=raw_request_body.get("title", "New Chat from " + db_user.id)
    )
    return user_identifier, project, conversation_id_from_owi, db_user

# Remove or update the old /test_ollama_prompt if it's now redundant
# class OllamaSimplePromptRequest(BaseModel): ...
# @app.post("/test_ollama_prompt", tags=["Ollama Tests"]) ...