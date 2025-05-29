# server/app/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import uuid
import time 
from typing import List, Dict, Optional, Union, Any
import logging # Import logging

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
from .logger_config import setup_logging # Import your logging setup function

logger = logging.getLogger(__name__) # Get a logger instance for this module

app = FastAPI(title="RAG Workspace Server")

# In-memory store for active project per user (user_identifier: project_id)
active_project_for_user: Dict[str, str] = {}

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    setup_logging() 
    logger.info("RAG Server on_event[startup] triggered.")

    llm_services.initialize_llm_clients() 
    
    # Attempt to pull default Ollama models
    # This assumes Ollama service is up or starting up due to depends_on
    try:
        logger.info("Attempting to ensure default Ollama models are pulled...")
        await llm_services.ensure_default_ollama_models()
        logger.info("Default Ollama model check/pull process complete.")
    except Exception as e:
        logger.error(f"Error during ensure_default_ollama_models: {e}", exc_info=True)

    create_db_and_tables()
    active_project_for_user.clear() 

    db: Session = SessionLocal()
    try:
        # ... (Default project and admin user linking logic as before) ...
        project_name = f"{settings.WORKSPACE_NAME} Project"; project = chat_service.get_or_create_project(db, project_name=project_name, description=f"Default project for workspace '{settings.WORKSPACE_NAME}'")
        admin_user_identifier = settings.WEBUI_ADMIN_USER_EMAIL if settings.WEBUI_ADMIN_USER_EMAIL else f"admin_{settings.WORKSPACE_NAME}"
        user = chat_service.get_or_create_user(db, user_identifier=admin_user_identifier) # Param name check
        chat_service.ensure_user_linked_to_project(db, user_id=user.id, project_id=project.id)
        logger.info(f"Default project '{project.name}' (ID: {project.id}) linked to user '{user.id}'.")

        if settings.AUTO_SEED_ON_STARTUP:
            # ... (seeding logic as before) ...
            if rag_service.is_rag_service_ready():
                logger.info(f"AUTO_SEED_ON_STARTUP true. Seeding for project '{project.id}'...")
                seed_dir = "/app/seed_data"; rag_service.seed_documents_from_directory(project_id=project.id, seed_dir_path=seed_dir)
            else: logger.warning("AUTO_SEED true, but RAG components not ready. Skipping seeding.")
    except Exception as e: 
        logger.error(f"Startup project/user/seed init error: {e}", exc_info=True)
    finally: 
        db.close()
        
    logger.info(f"RAG Server for '{settings.WORKSPACE_NAME}' started successfully.")

@app.on_event("shutdown")
async def shutdown_event():
    await llm_services.close_llm_clients() 
    logger.info(f"RAG Server for {settings.WORKSPACE_NAME} shutting down.")

# --- Health and Root Endpoints ---
@app.get("/health", tags=["Health Check"])
async def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "ok", "message": f"RAG Server for {settings.WORKSPACE_NAME} is healthy"}

@app.get("/", tags=["Root"])
async def read_root():
    logger.info("Root endpoint called.")
    return {"message": f"Welcome to {settings.WORKSPACE_NAME} RAG Server!"}

# --- OpenAI Compatible Endpoints (/v1) ---
@app.get("/v1/models", response_model=ModelList, tags=["OpenAI Compatibility"])
async def list_models():
    logger.info("Fetching available models for /v1/models endpoint.")
    available_models = []

    # Ollama models from settings
    if settings.DEFAULT_OLLAMA_MODELS_TO_PULL:
        for model_tag in settings.DEFAULT_OLLAMA_MODELS_TO_PULL:
            # We advertise them with the "ollama/" prefix for routing in chat_completions
            available_models.append(ModelCard(id=f"ollama/{model_tag}", owned_by="Gen AI Enable"))
        logger.debug(f"Advertising Ollama models: {settings.DEFAULT_OLLAMA_MODELS_TO_PULL}")
    else:
        logger.warning("No DEFAULT_OLLAMA_MODELS_TO_PULL configured in settings.")
        # Add a comment for developers on how to add more Ollama models:
        # To add more Ollama models to be advertised:
        # 1. Ensure they are pulled into your Ollama instance.
        # 2. Add their exact tags (e.g., "mistral:latest") to the
        #    DEFAULT_OLLAMA_MODELS_TO_PULL list in server/app/config.py or override via .env.

    # Gemini models
    if settings.GEMINI_API_KEY:
        # You can expand this list if you want to support more specific Gemini model versions
        # Ensure the model ID here matches what your gemini_chat_stream_generator expects
        # (e.g., "gemini-1.5-flash-latest" without the "gemini/" prefix for the SDK usually)
        available_models.append(ModelCard(id="gemini/gemini-1.5-flash-latest", owned_by="Gen AI Enable"))
        logger.debug("Advertising Gemini models as GEMINI_API_KEY is set.")
        # Comment for developers:
        # To add more Gemini models:
        # 1. Ensure your GEMINI_API_KEY has access to them.
        # 2. Add a ModelCard entry here (e.g., ModelCard(id="gemini/gemini-1.5-pro-latest", ...)).
        # 3. Update llm_services.py to handle the new model ID if specific logic is needed.
    else:
        logger.info("GEMINI_API_KEY not set. Gemini models will not be advertised.")


    # OpenAI models
    if settings.OPENAI_API_KEY and llm_services.openai_llm_client: # Check if client initialized
        available_models.append(ModelCard(id="openai/gpt-3.5-turbo", owned_by="Gen AI Enable"))
        available_models.append(ModelCard(id="openai/gpt-4o", owned_by="Gen AI Enable"))
        logger.debug("Advertising OpenAI models as OPENAI_API_KEY is set and client initialized.")
        # Comment for developers:
        # To add more OpenAI models:
        # 1. Ensure your OPENAI_API_KEY has access.
        # 2. Add a ModelCard entry here.
        # 3. Update llm_services.py if specific handling for the new model ID is needed.
    else:
        logger.info("OPENAI_API_KEY not set or client not initialized. OpenAI models will not be advertised.")
        
    logger.debug(f"Final list of models being advertised: {[m.id for m in available_models]}")
    return ModelList(data=available_models)

# --- Context Helper ---
async def _get_request_context(raw_request_body: dict, request_payload: ChatCompletionRequest, db: Session):
    global active_project_for_user # Allow access to the global

    user_identifier = "guest_user" 
    owi_variables = raw_request_body.get("variables", {})
    if owi_variables and isinstance(owi_variables, dict) and owi_variables.get("{{USER_NAME}}"):
        user_identifier = owi_variables["{{USER_NAME}}"]
    elif request_payload.user: 
        user_identifier = request_payload.user
    
    db_user = chat_service.get_or_create_user(db, user_identifier=user_identifier) # Param name updated

    current_project_id = active_project_for_user.get(db_user.id)
    project_obj: Optional[db_models.Project] = None

    if current_project_id:
        project_obj = db.query(db_models.Project).filter(db_models.Project.id == current_project_id).first()
        if not project_obj:
            logger.warning(f"Active project ID '{current_project_id}' for user '{db_user.id}' not found. Clearing active project.")
            active_project_for_user.pop(db_user.id, None)
            current_project_id = None 

    if not current_project_id: 
        default_project_name = f"{settings.WORKSPACE_NAME} Project"
        project_obj = chat_service.get_or_create_project(db, project_name=default_project_name)
        chat_service.ensure_user_linked_to_project(db, user_id=db_user.id, project_id=project_obj.id)
        active_project_for_user[db_user.id] = project_obj.id
        logger.info(f"User '{db_user.id}' defaulted to project '{project_obj.name}' (ID: {project_obj.id})")
    
    conversation_id_from_owi = raw_request_body.get("chat_id")
    if not conversation_id_from_owi: 
        conversation_id_from_owi = str(uuid.uuid4())
        logger.info(f"'chat_id' not found in OWI request for user '{db_user.id}'. Generated new backend conv ID: {conversation_id_from_owi}")
    
    chat_service.get_or_create_conversation(
        db, 
        conversation_id_from_owi=conversation_id_from_owi, 
        project_id=project_obj.id, 
        creator_user_id=db_user.id,
        title=raw_request_body.get("title", f"Chat in {project_obj.name}")
    )
    return db_user, project_obj, conversation_id_from_owi


# --- Main Chat Completions Endpoint ---
@app.post("/v1/chat/completions", tags=["OpenAI Compatibility"])
async def chat_completions(
    request: ChatCompletionRequest, 
    fastapi_request: FastAPIRequest,
    db: Session = Depends(get_db)
):
    global active_project_for_user

    raw_request_body = await fastapi_request.json()
    db_user, current_project_for_turn, conversation_id_from_owi = await _get_request_context(raw_request_body, request, db)
    logger.info(f"Processing chat completion for user '{db_user.id}' in project '{current_project_for_turn.name}' (ConvID: {conversation_id_from_owi}) Model: {request.model}")

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
            if current_user_query_content:
                 chat_service.add_message_to_database(db, conversation_id_from_owi, "user", current_user_query_content, author_user_id=db_user.id)
            
            new_project_obj = chat_service.get_or_create_project(db, project_name=new_project_name)
            chat_service.ensure_user_linked_to_project(db, user_id=db_user.id, project_id=new_project_obj.id)
            if rag_service.is_rag_service_ready(): # Ensure RAG service is ready before creating collection
                rag_service.get_or_create_project_collection(project_id=new_project_obj.id)
            else:
                logger.warning(f"RAG service not ready, cannot ensure ChromaDB collection for new project '{new_project_obj.name}'")
            
            active_project_for_user[db_user.id] = new_project_obj.id
            response_message = f"Switched to project: '{new_project_obj.name}'. RAG and chat are now scoped to this project."
            logger.info(f"User '{db_user.id}' switched project to '{new_project_obj.name}' (ID: {new_project_obj.id})")
            project_switched_successfully = True
        
        if not project_switched_successfully:
             logger.warning(f"Invalid !use_project command by user '{db_user.id}': {current_user_query_content}")
             # If command was invalid but there was content, save it as a user message
             if current_user_query_content and not (len(command_parts) > 1 and command_parts[1].strip()): # only if not saved above
                  chat_service.add_message_to_database(db, conversation_id_from_owi, "user", current_user_query_content, author_user_id=db_user.id)

        chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", response_message, author_user_id=None, model_used="system_command")
        
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

    # --- Regular message processing ---
    if current_user_query_content:
        chat_service.add_message_to_database(db, conversation_id_from_owi, "user", current_user_query_content, author_user_id=db_user.id)
    
    messages_for_llm_with_rag_context = list(original_user_messages)
    if current_user_query_content and rag_service.is_rag_service_ready(): 
        try:
            relevant_chunks = rag_service.query_project_collection(project_id=current_project_for_turn.id, query_text=current_user_query_content, n_results=settings.RAG_TOP_K)
            if relevant_chunks:
                retrieved_context_str = "\n\n--- Relevant Context Retrieved ---\n" + "\n\n".join(relevant_chunks) + "\n--- End of Context ---\n"
                context_system_message = ChatMessageInput(role="system", content=f"Based on the following context, answer the user's query:\n{retrieved_context_str}")
                if messages_for_llm_with_rag_context and messages_for_llm_with_rag_context[-1].role == "user":
                    messages_for_llm_with_rag_context.insert(len(messages_for_llm_with_rag_context) - 1, context_system_message)
                else: 
                    messages_for_llm_with_rag_context.insert(0, context_system_message)
                logger.info(f"RAG: Prepended {len(relevant_chunks)} context chunks for project '{current_project_for_turn.id}'.")
        except Exception as e: 
            logger.error(f"RAG: Failed to retrieve/inject context for project '{current_project_for_turn.id}': {e}", exc_info=True)

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

# Keep or remove /test_ollama_prompt as needed.
# If kept, ensure OllamaSimplePromptRequest Pydantic model is defined:
# from pydantic import BaseModel
# class OllamaSimplePromptRequest(BaseModel):
#     model: str = "gemma:2b"
#     prompt: str
#     stream: bool = False
# @app.post("/test_ollama_prompt", tags=["Ollama Tests"]) ... (existing logic)
