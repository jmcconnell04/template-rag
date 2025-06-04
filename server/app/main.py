# server/app/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import uuid
import time 
import asyncio 
from typing import List, Dict, Optional, Union, Any
import logging 
import json 

# Project-specific imports
from .config import settings # CORRECTED IMPORT for settings
from .openai_models import (
    ModelCard, ModelList, ChatCompletionRequest, ChatMessageInput,
    ChatCompletionStreamResponse, ChatCompletionStreamChoice, ChatCompletionStreamChoiceDelta,
    ChatCompletionResponse, ChatCompletionChoice, ChatMessageOutput, Usage
)
from .db.database import create_db_and_tables, get_db, SessionLocal
from .db import models as db_models
from .services import chat_service, rag_service
from .core import llm_services
from .logger_config import setup_logging # This should be correct if logger_config is in the same 'app' dir

logger = logging.getLogger(__name__) 

app = FastAPI(title="RAG Workspace Server")

active_project_for_user: Dict[str, str] = {}

async def _run_background_startup_tasks():
    # ... (content as before from response #70)
    try:
        logger.info("Background task: Attempting to ensure default Ollama models are pulled...")
        await llm_services.ensure_default_ollama_models_are_pulled_async() 
        logger.info("Background task: Default Ollama model check/pull process complete.")
    except Exception as e:
        logger.error(f"Background task error during ensure_default_ollama_models: {e}", exc_info=True)

    db_for_seed: Session = SessionLocal()
    try:
        if settings.AUTO_SEED_ON_STARTUP:
            project_name_for_seed = f"{settings.WORKSPACE_NAME} Project"
            project_for_seed = db_for_seed.query(db_models.Project).filter(db_models.Project.name == project_name_for_seed).first()
            if project_for_seed:
                if rag_service.is_rag_service_ready():
                    logger.info(f"Background task: AUTO_SEED_ON_STARTUP true. Seeding for project '{project_for_seed.id}'...")
                    seed_dir = "/app/seed_data" 
                    rag_service.seed_documents_from_directory(project_id=project_for_seed.id, seed_dir_path=seed_dir)
                else: 
                    logger.warning("Background task: AUTO_SEED true, but RAG components not ready. Skipping seeding.")
            else:
                logger.error(f"Background task: Default project '{project_name_for_seed}' not found for seeding.")
    except Exception as e:
        logger.error(f"Background task error during RAG seeding: {e}", exc_info=True)
    finally:
        db_for_seed.close()


@app.on_event("startup")
async def startup_event(): 
    setup_logging() 
    logger.info("RAG Server on_event[startup] triggered.")
    llm_services.initialize_llm_clients() 
    
    create_db_and_tables()
    active_project_for_user.clear() 

    db: Session = SessionLocal()
    try:
        project_name = f"{settings.WORKSPACE_NAME} Project"
        project = chat_service.get_or_create_project(db, project_name=project_name, description=f"Default project for workspace '{settings.WORKSPACE_NAME}'")
        
        admin_user_identifier = settings.WEBUI_ADMIN_USER_EMAIL
        if not admin_user_identifier: 
            admin_user_identifier = f"admin_{settings.WORKSPACE_NAME}"
        
        user = chat_service.get_or_create_user(db, user_identifier=admin_user_identifier)
        chat_service.ensure_user_linked_to_project(db, user_id=user.id, project_id=project.id)
        logger.info(f"Default project '{project.name}' (ID: {project.id}) linked to user '{user.id}'.")
    except Exception as e: 
        logger.error(f"Startup project/user init error: {e}", exc_info=True)
        # import traceback # Already imported if needed
        # traceback.print_exc() # Already in logger.error with exc_info=True
    finally: 
        db.close()
        
    asyncio.create_task(_run_background_startup_tasks())
    logger.info("Background tasks for model pulling and seeding have been scheduled.")
        
    logger.info(f"RAG Server for '{settings.WORKSPACE_NAME}' HTTP services starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    await llm_services.close_llm_clients() 
    logger.info(f"RAG Server for {settings.WORKSPACE_NAME} shutting down.")

@app.get("/health", tags=["Health Check"])
async def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "ok", "message": f"RAG Server for {settings.WORKSPACE_NAME} is healthy"}

@app.get("/", tags=["Root"])
async def read_root():
    logger.info("Root endpoint called.")
    return {"message": f"Welcome to {settings.WORKSPACE_NAME} RAG Server!"}

@app.get("/v1/models", response_model=ModelList, tags=["OpenAI Compatibility"])
async def list_models():
    # ... (content as before from response #70)
    logger.info("Fetching available models for /v1/models endpoint.")
    available_models = []
    if settings.DEFAULT_OLLAMA_MODELS_TO_PULL:
        for model_tag in settings.DEFAULT_OLLAMA_MODELS_TO_PULL:
            available_models.append(ModelCard(id=f"ollama/{model_tag}", owned_by="Gen AI Enable"))
    if settings.GEMINI_API_KEY: available_models.append(ModelCard(id="gemini/gemini-1.5-flash-latest", owned_by="Gen AI Enable"))
    if settings.OPENAI_API_KEY and llm_services.openai_llm_client: 
        available_models.append(ModelCard(id="openai/gpt-3.5-turbo", owned_by="Gen AI Enable"))
        available_models.append(ModelCard(id="openai/gpt-4o", owned_by="Gen AI Enable"))
    logger.debug(f"Models being advertised: {[m.id for m in available_models]}")
    return ModelList(data=available_models)

async def _get_request_context(raw_request_body: dict, request_payload: ChatCompletionRequest, db: Session):
    # ... (content as before from response #70, with enhanced logging for user ID)
    global active_project_for_user, declared_username_for_chat # Ensure declared_username_for_chat is defined if used
    
    logger.info("--- User Identification Attempt within _get_request_context ---")
    # logger.info(f"Full raw_request_body from OpenWebUI: {json.dumps(raw_request_body, indent=2, default=str)}")
    
    user_identifier = "guest_user" 
    
    conversation_id_for_user_lookup = raw_request_body.get("chat_id") # Use chat_id from request
    if conversation_id_for_user_lookup and conversation_id_for_user_lookup in declared_username_for_chat:
        user_identifier = declared_username_for_chat[conversation_id_for_user_lookup]
        logger.info(f"Attempt 1: Found declared username '{user_identifier}' for chat_id '{conversation_id_for_user_lookup}' from !iam command.")
    else:
        if hasattr(request_payload, 'user') and request_payload.user:
            user_identifier = request_payload.user
            logger.info(f"Attempt 2: Found user_identifier from standard 'request_payload.user': '{user_identifier}'")
        else:
            owi_variables = raw_request_body.get("variables")
            if owi_variables and isinstance(owi_variables, dict):
                logger.debug(f"Attempt 3: OpenWebUI 'variables' block found: {json.dumps(owi_variables, indent=2, default=str)}")
                user_name_from_vars_exact = owi_variables.get("{{USER_NAME}}")
                if user_name_from_vars_exact:
                    user_identifier = user_name_from_vars_exact
                    logger.info(f"Found user_identifier from owi_variables.{{USER_NAME}}: '{user_identifier}'")
                else: # Fallback for other keys
                    potential_keys = ["USER_NAME", "username", "user_name", "name", "id", "email"]
                    found_in_potential = False
                    for key in potential_keys:
                        if owi_variables.get(key): user_identifier = owi_variables.get(key); logger.info(f"Found user_identifier from owi_variables.{key}: '{user_identifier}'"); found_in_potential = True; break
                    if not found_in_potential: logger.warning(f"Could not find known user name/ID key in owi_variables. Keys: {list(owi_variables.keys())}")
            if user_identifier == "guest_user": logger.warning(f"User ID ultimately defaulted to '{user_identifier}'.")
    
    db_user = chat_service.get_or_create_user(db, user_identifier=user_identifier)
    logger.info(f"Context established for DB user_id: '{db_user.id}' (derived from identifier: '{user_identifier}')")

    current_project_id = active_project_for_user.get(db_user.id); project_obj: Optional[db_models.Project] = None
    if current_project_id:
        project_obj = db.query(db_models.Project).filter(db_models.Project.id == current_project_id).first()
        if not project_obj: logger.warning(f"Active project ID '{current_project_id}' for user '{db_user.id}' not found. Clearing."); active_project_for_user.pop(db_user.id, None); current_project_id = None 
    if not current_project_id: 
        default_project_name = f"{settings.WORKSPACE_NAME} Project"; project_obj = chat_service.get_or_create_project(db, project_name=default_project_name)
        chat_service.ensure_user_linked_to_project(db, user_id=db_user.id, project_id=project_obj.id); active_project_for_user[db_user.id] = project_obj.id
        logger.info(f"User '{db_user.id}' defaulted to project: '{project_obj.name}' (ID: {project_obj.id})")
    
    final_conversation_id = conversation_id_for_user_lookup # Use the one we got from request
    if not final_conversation_id: 
        final_conversation_id = str(uuid.uuid4())
        logger.info(f"'chat_id' not found in OWI request. Generated new backend conv ID: {final_conversation_id} for user '{db_user.id}'")
    
    chat_service.get_or_create_conversation(db, conversation_id_from_owi=final_conversation_id, project_id=project_obj.id, creator_user_id=db_user.id, title=raw_request_body.get("title", f"Chat in {project_obj.name}"))
    return db_user, project_obj, final_conversation_id


@app.post("/v1/chat/completions", tags=["OpenAI Compatibility"])
async def chat_completions(request: ChatCompletionRequest, fastapi_request: FastAPIRequest, db: Session = Depends(get_db)):
    # ... (content as before from response #70, with !iam command logic using declared_username_for_chat)
    global active_project_for_user, declared_username_for_chat 
    raw_request_body = await fastapi_request.json() 
    db_user, current_project_for_turn, conversation_id_from_owi = await _get_request_context(raw_request_body, request, db)
    logger.info(f"Processing chat completion for user '{db_user.id}' (identified as: '{db_user.id}') in project '{current_project_for_turn.name}' (ConvID: {conversation_id_from_owi}) Model: {request.model}")
    original_user_messages = [ChatMessageInput(role=m.role, content=m.content) for m in request.messages]; current_user_query_content = ""
    if original_user_messages and original_user_messages[-1].role == "user": current_user_query_content = original_user_messages[-1].content
    request_id = f"chatcmpl-{uuid.uuid4()}"

    # --- Handle `!iam <username>` command ---
    if current_user_query_content.lower().startswith("!iam "):
        command_parts = current_user_query_content.split(" ", 1)
        response_message = "Invalid command. Usage: `!iam <YourName>`"
        if len(command_parts) > 1 and command_parts[1].strip():
            declared_name = command_parts[1].strip()
            declared_username_for_chat[conversation_id_from_owi] = declared_name # Store for this chat_id
            # Log the original command message with the newly declared name
            chat_service.add_message_to_database(db, conversation_id_from_owi, "user", current_user_query_content, author_user_id=declared_name)
            response_message = f"OK, I will identify you as '{declared_name}' for this chat session (ID: {conversation_id_from_owi})."
            logger.info(f"User declared identity as '{declared_name}' for chat_id '{conversation_id_from_owi}'.")
        else: 
             if current_user_query_content: chat_service.add_message_to_database(db, conversation_id_from_owi, "user", current_user_query_content, author_user_id=db_user.id)
        chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", response_message, author_user_id=None, model_used="system_command")
        if request.stream:
            async def cmd_stream(): 
                yield f"data: {ChatCompletionStreamResponse(id=request_id, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(content=response_message, role='assistant'))]).model_dump_json()}\n\n"
                yield f"data: {ChatCompletionStreamResponse(id=request_id, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(), finish_reason='stop')]).model_dump_json()}\n\n"
                yield f"data: [DONE]\n\n"
            return StreamingResponse(cmd_stream(), media_type="text/event-stream")
        else: return ChatCompletionResponse(id=request_id, model=request.model, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=response_message), finish_reason="stop")], usage=Usage(prompt_tokens=0, completion_tokens=len(response_message)//4 if response_message else 0, total_tokens=len(response_message)//4 if response_message else 0))
    
    # --- Handle `!use_project` command ---
    elif current_user_query_content.lower().startswith("!use_project "):
        # ... (command handling logic as before from response #68, ensure author_user_id for user message uses db_user.id) ...
        command_parts = current_user_query_content.split(" ", 1); response_message = "Invalid command. Usage: !use_project <project_name>"; project_switched_successfully = False
        if len(command_parts) > 1 and command_parts[1].strip():
            new_project_name = command_parts[1].strip()
            if current_user_query_content: chat_service.add_message_to_database(db, conversation_id_from_owi, "user", current_user_query_content, author_user_id=db_user.id)
            new_project_obj = chat_service.get_or_create_project(db, project_name=new_project_name)
            chat_service.ensure_user_linked_to_project(db, user_id=db_user.id, project_id=new_project_obj.id)
            if rag_service.is_rag_service_ready(): rag_service.get_or_create_project_collection(project_id=new_project_obj.id)
            else: logger.warning(f"RAG service not ready, cannot ensure ChromaDB collection for new project '{new_project_obj.name}'")
            active_project_for_user[db_user.id] = new_project_obj.id; response_message = f"Switched to project: '{new_project_obj.name}'."; logger.info(f"User '{db_user.id}' switched project to '{new_project_obj.name}'")
            project_switched_successfully = True
        if not project_switched_successfully: logger.warning(f"Invalid !use_project command by user '{db_user.id}': {current_user_query_content}")
        if current_user_query_content and not project_switched_successfully and not (len(command_parts) > 1 and command_parts[1].strip()): chat_service.add_message_to_database(db, conversation_id_from_owi, "user", current_user_query_content, author_user_id=db_user.id)
        chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", response_message, author_user_id=None, model_used="system_command")
        if request.stream:
            async def cmd_stream_project(): 
                yield f"data: {ChatCompletionStreamResponse(id=request_id, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(content=response_message, role='assistant'))]).model_dump_json()}\n\n"
                yield f"data: {ChatCompletionStreamResponse(id=request_id, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(), finish_reason='stop')]).model_dump_json()}\n\n"
                yield f"data: [DONE]\n\n"
            return StreamingResponse(cmd_stream_project(), media_type="text/event-stream")
        else: return ChatCompletionResponse(id=request_id, model=request.model, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=response_message), finish_reason="stop")], usage=Usage(prompt_tokens=0, completion_tokens=len(response_message)//4 if response_message else 0, total_tokens=len(response_message)//4 if response_message else 0))

    # --- Regular message processing ---
    if current_user_query_content: # Save user message if it wasn't a command
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
                else: messages_for_llm_with_rag_context.insert(0, context_system_message)
                logger.info(f"RAG: Prepended {len(relevant_chunks)} context chunks for project '{current_project_for_turn.id}'.")
        except Exception as e: logger.error(f"RAG: Failed to retrieve/inject context for project '{current_project_for_turn.id}': {e}", exc_info=True)

    if request.stream:
        return StreamingResponse(
            llm_services.route_chat_to_llm_stream(
                db=db, project_obj=current_project_for_turn, user_identifier=db_user.id, 
                conversation_id_from_owi=conversation_id_from_owi, model_id_from_request=request.model,
                messages_for_llm=messages_for_llm_with_rag_context, request_id=request_id
            ), media_type="text/event-stream"
        )
    else: 
        response_payload: ChatCompletionResponse = await llm_services.route_chat_to_llm_non_stream(
            db=db, project_obj=current_project_for_turn, user_identifier=db_user.id, 
            conversation_id_from_owi=conversation_id_from_owi, model_id_from_request=request.model,
            messages_for_llm=messages_for_llm_with_rag_context, request_id=request_id
        )
        return response_payload

# (OllamaSimplePromptRequest class and /test_ollama_prompt endpoint if kept)
