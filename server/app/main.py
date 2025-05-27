# server/app/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import uuid
import time # Ensure time is imported if used in Pydantic models via default_factory
from typing import List, Dict, Optional, Union, Any
import logging # Add this line

# Project-specific imports
from .config import settings
from .openai_models import (
    ModelCard, ModelList, ChatCompletionRequest, ChatMessageInput,
    ChatCompletionStreamResponse, ChatCompletionStreamChoice, ChatCompletionStreamChoiceDelta,
    ChatCompletionResponse, ChatCompletionChoice, ChatMessageOutput, Usage
)
from .db.database import create_db_and_tables, get_db, SessionLocal
from .db import models as db_models
from .services import chat_service, rag_service
from .core import llm_services
from .logger_config import setup_logging

logger = logging.getLogger(__name__) # Add this line

app = FastAPI(title="RAG Workspace Server")

# In-memory store for active project per user (user_identifier: project_id)
# NOTE: This is for single-instance demo purposes. Not suitable for production.
active_project_for_user: Dict[str, str] = {}

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    setup_logging() # Add this line to configure logging
    llm_services.initialize_llm_clients() # Initialize LLM clients via the service
    
    logger.info("Starting up RAG Server...") # Changed from print
    create_db_and_tables()
    active_project_for_user.clear() # Clear on startup for demo simplicity

    db: Session = SessionLocal()
    try:
        project_name = f"{settings.WORKSPACE_NAME} Project"
        project = chat_service.get_or_create_project(db, project_name=project_name, description=f"Default project for workspace '{settings.WORKSPACE_NAME}'")
        
        admin_user_identifier = settings.WEBUI_ADMIN_USER_EMAIL
        if not admin_user_identifier: 
            admin_user_identifier = f"admin_{settings.WORKSPACE_NAME}"
        
        user = chat_service.get_or_create_user(db, user_identifier=admin_user_identifier)
        chat_service.ensure_user_linked_to_project(db, user_id=user.id, project_id=project.id)
        logger.info(f"Default project '{project.name}' (ID: {project.id}) linked to user '{user.id}'.") # Changed from print

        if settings.AUTO_SEED_ON_STARTUP:
            if rag_service.CHROMA_CLIENT and rag_service.SENTENCE_TRANSFORMER_EF: # Check RAG components
                logger.info(f"AUTO_SEED_ON_STARTUP true. Seeding for project '{project.id}'...") # Changed from print
                seed_dir = "/app/seed_data" # Mounted from ./rag_files/seed/
                rag_service.seed_documents_from_directory(project_id=project.id, seed_dir_path=seed_dir)
            else: 
                logger.warning("AUTO_SEED_ON_STARTUP true, but RAG service (Chroma/Embeddings) not ready. Skipping seeding.") # Changed from print
    except Exception as e: 
        logger.exception(f"Startup project/user/seed init error: {e}") # Changed from print and traceback.print_exc()
        # import traceback # Removed this line
        # traceback.print_exc() # Removed this line
    finally: 
        db.close()
        
    logger.info(f"RAG Server for '{settings.WORKSPACE_NAME}' started successfully.") # Changed from print

@app.on_event("shutdown")
async def shutdown_event():
    await llm_services.close_llm_clients() # Close LLM clients via the service
    logger.info(f"RAG Server for {settings.WORKSPACE_NAME} shutting down.") # Changed from print

# --- Health and Root Endpoints ---
@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok", "message": f"RAG Server for {settings.WORKSPACE_NAME} is healthy"}

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to {settings.WORKSPACE_NAME} RAG Server!"}

# --- OpenAI Compatible Endpoints (/v1) ---
@app.get("/v1/models", response_model=ModelList, tags=["OpenAI Compatibility"])
async def list_models():
    available_models = []
    # Define which Ollama models your server will advertise
    ollama_models_to_advertise = ["ollama/gemma:2b", "ollama/qwen:1.8b", "ollama/llama3:70b"] # Your specified models
    for model_id in ollama_models_to_advertise:
        available_models.append(ModelCard(id=model_id, owned_by="Gen AI Enable"))

    if settings.GEMINI_API_KEY:
        available_models.append(ModelCard(id="gemini/gemini-1.5-flash-latest", owned_by="Gen AI Enable"))
        # Add other Gemini model IDs here if you want to offer them
        # available_models.append(ModelCard(id="gemini/gemini-1.5-pro-latest", owned_by="Gen AI Enable"))
    if settings.OPENAI_API_KEY: # No need to check openai_client here, service will handle if not init
        available_models.append(ModelCard(id="openai/gpt-3.5-turbo", owned_by="Gen AI Enable"))
        available_models.append(ModelCard(id="openai/gpt-4o", owned_by="Gen AI Enable"))
        
    return ModelList(data=available_models)

# --- Context Helper ---
async def _get_request_context(raw_request_body: dict, request_payload: ChatCompletionRequest, db: Session):
    global active_project_for_user

    user_identifier = "guest_user" 
    owi_variables = raw_request_body.get("variables", {})
    if owi_variables and isinstance(owi_variables, dict) and owi_variables.get("{{USER_NAME}}"):
        user_identifier = owi_variables["{{USER_NAME}}"]
    elif request_payload.user: 
        user_identifier = request_payload.user
    
    db_user = chat_service.get_or_create_user(db, user_identifier=user_identifier)

    current_project_id = active_project_for_user.get(db_user.id)
    project_obj: Optional[db_models.Project] = None

    if current_project_id:
        project_obj = db.query(db_models.Project).filter(db_models.Project.id == current_project_id).first()
        if not project_obj:
            active_project_for_user.pop(db_user.id, None)
            current_project_id = None # Force fallback to default

    if not current_project_id: 
        default_project_name = f"{settings.WORKSPACE_NAME} Project"
        project_obj = chat_service.get_or_create_project(db, project_name=default_project_name)
        chat_service.ensure_user_linked_to_project(db, user_id=db_user.id, project_id=project_obj.id)
        active_project_for_user[db_user.id] = project_obj.id
    
    conversation_id_from_owi = raw_request_body.get("chat_id")
    if not conversation_id_from_owi: # Should ideally always be sent by OWI for existing or new chats
        conversation_id_from_owi = str(uuid.uuid4())
        logger.info(f"'chat_id' not found in OWI request for user '{db_user.id}'. Generated new backend conv ID: {conversation_id_from_owi}") # Changed from print
    
    chat_service.get_or_create_conversation(
        db, 
        conversation_id_from_owi=conversation_id_from_owi, 
        project_id=project_obj.id, 
        creator_user_id=db_user.id,
        title=raw_request_body.get("title", f"Chat in {project_obj.name}") # OWI might send a title
    )
    return db_user, project_obj, conversation_id_from_owi


# --- Main Chat Completions Endpoint ---
@app.post("/v1/chat/completions", tags=["OpenAI Compatibility"])
async def chat_completions(
    request: ChatCompletionRequest, 
    fastapi_request: FastAPIRequest,
    db: Session = Depends(get_db)
):
    global active_project_for_user # To modify for !use_project

    raw_request_body = await fastapi_request.json()
    # current_project_for_turn is the project active *before* this message is processed
    db_user, current_project_for_turn, conversation_id_from_owi = await _get_request_context(raw_request_body, request, db)
    
    original_user_messages = [ChatMessageInput(role=m.role, content=m.content) for m in request.messages]
    current_user_query_content = ""
    if original_user_messages and original_user_messages[-1].role == "user":
        current_user_query_content = original_user_messages[-1].content

    request_id = f"chatcmpl-{uuid.uuid4()}"

    # --- Handle `!use_project` command ---
    if current_user_query_content.lower().startswith("!use_project "):
        command_parts = current_user_query_content.split(" ", 1)
        response_message = "Invalid command. Usage: !use_project <project_name>"
        project_switched_successfully = False
        if len(command_parts) > 1 and command_parts[1].strip():
            new_project_name = command_parts[1].strip()
            # Save the command message to the current conversation
            if current_user_query_content: # Should always be true here
                 chat_service.add_message_to_database(db, conversation_id_from_owi, "user", current_user_query_content, author_user_id=db_user.id)
            
            new_project_obj = chat_service.get_or_create_project(db, project_name=new_project_name)
            chat_service.ensure_user_linked_to_project(db, user_id=db_user.id, project_id=new_project_obj.id)
            rag_service.get_or_create_project_collection(project_id=new_project_obj.id) # Ensure Chroma collection exists
            
            active_project_for_user[db_user.id] = new_project_obj.id # Update active project
            response_message = f"Switched to project: '{new_project_obj.name}'. RAG and chat are now scoped to this project."
            logger.info(f"User '{db_user.id}' switched project to '{new_project_obj.name}' (ID: {new_project_obj.id})") # Changed from print
            project_switched_successfully = True
        
        if not project_switched_successfully:
             logger.warning(f"Invalid !use_project command by user '{db_user.id}': {current_user_query_content}") # Changed from print
             if current_user_query_content: # Still save the invalid command attempt
                  chat_service.add_message_to_database(db, conversation_id_from_owi, "user", current_user_query_content, author_user_id=db_user.id)


        # Save system response to the current conversation
        chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", response_message, author_user_id=None, model_used="system_command")
        
        # Return confirmation to OpenWebUI
        if request.stream:
            async def cmd_stream():
                chunk = ChatCompletionStreamResponse(id=request_id, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(content=response_message, role='assistant'))])
                yield f"data: {chunk.model_dump_json()}\n\n"
                finish_chunk = ChatCompletionStreamResponse(id=request_id, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(), finish_reason='stop')])
                yield f"data: {finish_chunk.model_dump_json()}\n\n"
                yield f"data: [DONE]\n\n"
            return StreamingResponse(cmd_stream(), media_type="text/event-stream")
        else:
            return ChatCompletionResponse(id=request_id, model=request.model, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=response_message), finish_reason="stop")], usage=Usage(prompt_tokens=0, completion_tokens=len(response_message)//4 if response_message else 0, total_tokens=len(response_message)//4 if response_message else 0))

    # --- Regular message processing (if not a command) ---
    if current_user_query_content: # Save user message if it wasn't a command (or was an invalid one that fell through)
        chat_service.add_message_to_database(db, conversation_id_from_owi, "user", current_user_query_content, author_user_id=db_user.id)
    
    messages_for_llm_with_rag_context = list(original_user_messages) # Start with original messages from OWI for this turn
    if current_user_query_content: 
        try:
            # Use current_project_for_turn (which is the active project for this request)
            relevant_chunks = rag_service.query_project_collection(project_id=current_project_for_turn.id, query_text=current_user_query_content, n_results=settings.RAG_TOP_K)
            if relevant_chunks:
                retrieved_context_str = "\n\n--- Relevant Context Retrieved ---\n" + "\n\n".join(relevant_chunks) + "\n--- End of Context ---\n"
                context_system_message = ChatMessageInput(role="system", content=f"Based on the following context, answer the user's query:\n{retrieved_context_str}")
                
                # Insert system context message before the last user message
                if messages_for_llm_with_rag_context and messages_for_llm_with_rag_context[-1].role == "user":
                    messages_for_llm_with_rag_context.insert(len(messages_for_llm_with_rag_context) - 1, context_system_message)
                else: # If no user message or list is empty for some reason, just prepend
                    messages_for_llm_with_rag_context.insert(0, context_system_message)
                logger.info(f"RAG: Prepended {len(relevant_chunks)} context chunks for project '{current_project_for_turn.id}'.") # Changed from print
        except Exception as e: 
            logger.error(f"RAG: Failed to retrieve/inject context for project '{current_project_for_turn.id}': {e}") # Changed from print

    # --- Call the Centralized LLM Service ---
    if request.stream:
        return StreamingResponse(
            llm_services.route_chat_to_llm_stream(
                db=db, project_obj=current_project_for_turn, user_identifier=db_user.id,
                conversation_id_from_owi=conversation_id_from_owi, model_id_from_request=request.model,
                messages_for_llm=messages_for_llm_with_rag_context, request_id=request_id
            ),
            media_type="text/event-stream"
        )
    else: # Non-streaming
        response_payload: ChatCompletionResponse = await llm_services.route_chat_to_llm_non_stream(
            db=db, project_obj=current_project_for_turn, user_identifier=db_user.id,
            conversation_id_from_owi=conversation_id_from_owi, model_id_from_request=request.model,
            messages_for_llm=messages_for_llm_with_rag_context, request_id=request_id
        )
        return response_payload

# Keep /test_ollama_prompt and its Pydantic model if you still use it for direct testing.
# For brevity, not re-pasting it here, but ensure it's defined if you need it.
# class OllamaSimplePromptRequest(BaseModel): ...
# @app.post("/test_ollama_prompt", tags=["Ollama Tests"]) ...