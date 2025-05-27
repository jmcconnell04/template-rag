# server/app/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel 
import httpx
import json
import time
import uuid
from typing import List, Dict, Optional, Union, Any # Ensure Any is imported

# SDKs for Cloud LLMs
import google.generativeai as genai
from openai import AsyncOpenAI

from .config import settings
from .openai_models import (
    ModelCard, ModelList, ChatCompletionRequest, ChatMessageInput,
    ChatCompletionStreamResponse, ChatCompletionStreamChoice, ChatCompletionStreamChoiceDelta,
    ChatCompletionResponse, ChatCompletionChoice, ChatMessageOutput, Usage
)
from .db.database import create_db_and_tables, get_db, SessionLocal
from .db import models as db_models # To distinguish from Pydantic models
from .services import chat_service, rag_service

app = FastAPI(title="RAG Workspace Server")

# --- Global Variables & Clients ---
http_client = httpx.AsyncClient() # For Ollama
openai_client: Optional[AsyncOpenAI] = None # For OpenAI API

# In-memory store for active project per user (user_identifier: project_id)
# NOTE: This is for single-instance demo purposes. Not suitable for production.
active_project_for_user: Dict[str, str] = {}

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    global http_client, openai_client, active_project_for_user
    if http_client.is_closed: # Reinitialize if closed (e.g., after tests)
        http_client = httpx.AsyncClient()
    
    if settings.OPENAI_API_KEY:
        openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        print("INFO:     OpenAI client initialized.")
    else:
        print("INFO:     OpenAI API key not found. OpenAI models will be unavailable via this server.")

    if settings.GEMINI_API_KEY:
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            print("INFO:     Gemini client configured.")
        except Exception as e:
            print(f"ERROR:    Failed to configure Gemini client: {e}")
    else:
        print("INFO:     Gemini API key not found. Gemini models will be unavailable via this server.")

    print("INFO:     Starting up RAG Server...")
    create_db_and_tables()
    
    active_project_for_user = {} # Initialize/clear on startup for demo simplicity

    db: Session = SessionLocal()
    try:
        project_name = f"{settings.WORKSPACE_NAME} Project"
        project = chat_service.get_or_create_project(db, project_name=project_name, description=f"Default project for workspace '{settings.WORKSPACE_NAME}'")
        
        admin_user_identifier = settings.WEBUI_ADMIN_USER_EMAIL
        if not admin_user_identifier: 
            admin_user_identifier = f"admin_{settings.WORKSPACE_NAME}"
        
        user = chat_service.get_or_create_user(db, user_identifier=admin_user_identifier)
        chat_service.ensure_user_linked_to_project(db, user_id=user.id, project_id=project.id)
        print(f"INFO:     Default project '{project.name}' (ID: {project.id}) checked/created and linked to user '{user.id}'.")

        if settings.AUTO_SEED_ON_STARTUP:
            if rag_service.CHROMA_CLIENT and rag_service.SENTENCE_TRANSFORMER_EF:
                print(f"INFO:     AUTO_SEED_ON_STARTUP is true. Seeding documents for project '{project.id}'...")
                seed_dir = "/app/seed_data" # Mounted from ./rag_files/seed/
                rag_service.seed_documents_from_directory(project_id=project.id, seed_dir_path=seed_dir)
            else:
                print("WARNING:  AUTO_SEED_ON_STARTUP true, but RAG service components not ready. Skipping seeding.")
    except Exception as e:
        print(f"ERROR:    Error during startup project/user/seed initialization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
        
    print(f"INFO:     RAG Server for '{settings.WORKSPACE_NAME}' started successfully.")

@app.on_event("shutdown")
async def shutdown_event():
    global openai_client
    await http_client.aclose()
    if openai_client:
        await openai_client.close()
    print(f"INFO:     RAG Server for {settings.WORKSPACE_NAME} shutting down.")

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
    ollama_models_to_advertise = ["ollama/gemma:2b", "ollama/qwen:1.8b", "ollama/llama3:70b"] # From your spec
    for model_id in ollama_models_to_advertise:
        available_models.append(ModelCard(id=model_id, owned_by="Gen AI Enable"))

    if settings.GEMINI_API_KEY:
        available_models.append(ModelCard(id="gemini/gemini-1.5-flash-latest", owned_by="Gen AI Enable"))
    if settings.OPENAI_API_KEY and openai_client:
        available_models.append(ModelCard(id="openai/gpt-3.5-turbo", owned_by="Gen AI Enable"))
        available_models.append(ModelCard(id="openai/gpt-4o", owned_by="Gen AI Enable"))
        
    return ModelList(data=available_models)

# --- Context Helper ---
async def _get_request_context(raw_request_body: dict, request_payload: ChatCompletionRequest, db: Session):
    global active_project_for_user

    user_identifier = "guest_user" # Default
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
            current_project_id = None

    if not current_project_id:
        default_project_name = f"{settings.WORKSPACE_NAME} Project"
        project_obj = chat_service.get_or_create_project(db, project_name=default_project_name)
        chat_service.ensure_user_linked_to_project(db, user_id=db_user.id, project_id=project_obj.id)
        active_project_for_user[db_user.id] = project_obj.id
    
    conversation_id_from_owi = raw_request_body.get("chat_id")
    if not conversation_id_from_owi:
        conversation_id_from_owi = str(uuid.uuid4())
    
    chat_service.get_or_create_conversation(
        db, conversation_id_from_owi=conversation_id_from_owi, 
        project_id=project_obj.id, creator_user_id=db_user.id,
        title=raw_request_body.get("title", f"Chat in {project_obj.name}")
    )
    return db_user, project_obj, conversation_id_from_owi

# --- LLM Stream Generators ---
async def ollama_chat_stream_generator(db: Session, project_obj: db_models.Project, user_identifier: str, conversation_id_from_owi: str, model_id: str, messages_for_llm: List[ChatMessageInput], request_id: str):
    ollama_model_name = model_id.replace("ollama/", "")
    ollama_formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm]
    payload = {"model": ollama_model_name, "messages": ollama_formatted_messages, "stream": True}
    assistant_response_buffer = ""
    try:
        async with http_client.stream("POST", f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=120.0) as response:
            if response.status_code != 200:
                error_content = await response.aread(); error_msg = f"Ollama API Error ({response.status_code}): {error_content.decode()}"
                print(f"ERROR: Stream - {error_msg}")
                yield f"data: {ChatCompletionStreamResponse(id=request_id,model=model_id,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f'Error: {error_msg}'),finish_reason='error')]).model_dump_json()}\n\n"
            else:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            delta_content = chunk.get("message", {}).get("content", "")
                            if delta_content is not None: assistant_response_buffer += delta_content
                            delta_role = "assistant" if chunk.get("message",{}).get("role") == "assistant" and delta_content and len(assistant_response_buffer.strip()) == len(delta_content.strip()) else None
                            stream_choice = ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=delta_content if delta_content else None, role=delta_role),finish_reason="stop" if chunk.get("done") else None)
                            yield f"data: {ChatCompletionStreamResponse(id=request_id, model=model_id, choices=[stream_choice]).model_dump_json()}\n\n"
                            if chunk.get("done"): break
                        except json.JSONDecodeError: print(f"WARNING: Ollama stream JSON decode error: {line}")
        if assistant_response_buffer and (getattr(response, 'status_code', 500) == 200):
            chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", assistant_response_buffer, author_user_id=None, model_used=model_id)
    except httpx.RequestError as e:
        error_msg = f"Request error (Ollama stream): {e}"; print(f"ERROR: {error_msg}")
        yield f"data: {ChatCompletionStreamResponse(id=request_id,model=model_id,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f'Error: {error_msg}'),finish_reason='error')]).model_dump_json()}\n\n"
    except Exception as e:
        error_msg = f"Generic error (Ollama stream): {e}"; print(f"ERROR: {error_msg}")
        yield f"data: {ChatCompletionStreamResponse(id=request_id,model=model_id,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f'Error: {error_msg}'),finish_reason='error')]).model_dump_json()}\n\n"
    finally: yield f"data: [DONE]\n\n"

async def gemini_chat_stream_generator(db: Session, project_obj: db_models.Project, user_identifier: str, conversation_id_from_owi: str, model_id_from_request: str, messages_for_llm: List[ChatMessageInput], request_id: str):
    if not settings.GEMINI_API_KEY:
        error_msg="Gemini API key not configured."; print(f"ERROR: Stream - {error_msg}")
        yield f"data: {ChatCompletionStreamResponse(id=request_id,model=model_id_from_request,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f'Error: {error_msg}'),finish_reason='error')]).model_dump_json()}\n\n"
        yield f"data: [DONE]\n\n"; return
    try:
        actual_gemini_model_name = model_id_from_request.split('/', 1)[1] if '/' in model_id_from_request else model_id_from_request
        if actual_gemini_model_name not in ["gemini-pro", "gemini-1.0-pro", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"] and not actual_gemini_model_name.startswith("models/"):
            actual_gemini_model_name = f"models/{actual_gemini_model_name}" # Common for newer specific versions

        gemini_history_contents = []; system_instruction = None
        temp_messages = list(messages_for_llm)
        if temp_messages and temp_messages[0].role == "system": system_instruction = temp_messages.pop(0).content
        for msg in temp_messages: gemini_history_contents.append({'role': "model" if msg.role == "assistant" else msg.role, 'parts': [{'text': msg.content}]})
        
        model_init_args = {}
        if system_instruction: model_init_args['system_instruction'] = system_instruction
        gemini_sdk_model = genai.GenerativeModel(actual_gemini_model_name, **model_init_args)
        
        response_stream = await gemini_sdk_model.generate_content_async(contents=gemini_history_contents, stream=True)
        assistant_response_buffer = ""
        first_chunk = True
        async for chunk in response_stream:
            delta_content = ""; finish_reason_gemini = None
            if hasattr(chunk, 'text') and chunk.text: delta_content = chunk.text
            elif hasattr(chunk, 'parts') and chunk.parts: 
                for part in chunk.parts: 
                    if hasattr(part, 'text'): delta_content += part.text
            
            # Check for finish reason (Google specific)
            # if hasattr(chunk, 'candidates') and chunk.candidates:
            #    if hasattr(chunk.candidates[0], 'finish_reason') and chunk.candidates[0].finish_reason:
            #        finish_reason_gemini = str(chunk.candidates[0].finish_reason).lower() # FINISH_REASON_STOP, FINISH_REASON_MAX_TOKENS etc.
            #        if "stop" in finish_reason_gemini: finish_reason_gemini = "stop"
            #        elif "max_tokens" in finish_reason_gemini: finish_reason_gemini = "length"
            
            if delta_content: assistant_response_buffer += delta_content
            
            delta_role_to_send = "assistant" if first_chunk and delta_content else None
            if delta_content: first_chunk = False

            stream_choice = ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(content=delta_content if delta_content else None, role=delta_role_to_send), finish_reason=finish_reason_gemini)
            yield f"data: {ChatCompletionStreamResponse(id=request_id, model=model_id_from_request, choices=[stream_choice]).model_dump_json()}\n\n"
        
        # After loop, ensure final "stop" is sent if not already implied by Gemini's stream
        if not finish_reason_gemini or finish_reason_gemini != "stop":
            yield f"data: {ChatCompletionStreamResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(), finish_reason='stop')]).model_dump_json()}\n\n"

        if assistant_response_buffer:
            chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", assistant_response_buffer, author_user_id=None, model_used=model_id_from_request)
    except Exception as e:
        error_msg = f"Error streaming from Gemini ({model_id_from_request}): {str(e)}"; print(f"ERROR: Stream - {error_msg}")
        yield f"data: {ChatCompletionStreamResponse(id=request_id,model=model_id_from_request,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f'Error: {error_msg}'),finish_reason='error')]).model_dump_json()}\n\n"
    finally: yield f"data: [DONE]\n\n"

async def openai_api_chat_stream_generator(db: Session, project_obj: db_models.Project, user_identifier: str, conversation_id_from_owi: str, model_id_from_request: str, messages_for_llm: List[ChatMessageInput], request_id: str):
    global openai_client
    if not settings.OPENAI_API_KEY or not openai_client:
        error_msg="OpenAI API key not configured/client not init."; print(f"ERROR: Stream - {error_msg}")
        yield f"data: {ChatCompletionStreamResponse(id=request_id,model=model_id_from_request,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f'Error: {error_msg}'),finish_reason='error')]).model_dump_json()}\n\n"
        yield f"data: [DONE]\n\n"; return
    try:
        actual_openai_model_name = model_id_from_request.split('/', 1)[1] if '/' in model_id_from_request else model_id_from_request
        openai_api_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm]
        assistant_response_buffer = ""
        stream = await openai_client.chat.completions.create(model=actual_openai_model_name, messages=openai_api_messages, stream=True)
        async for chunk in stream:
            delta_content = chunk.choices[0].delta.content
            role = chunk.choices[0].delta.role 
            if delta_content: assistant_response_buffer += delta_content
            stream_choice = ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(content=delta_content if delta_content else None, role=role if role else None), finish_reason=chunk.choices[0].finish_reason)
            yield f"data: {ChatCompletionStreamResponse(id=request_id, model=model_id_from_request, choices=[stream_choice]).model_dump_json()}\n\n"
            if chunk.choices[0].finish_reason: break
        if assistant_response_buffer:
             chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", assistant_response_buffer, author_user_id=None, model_used=model_id_from_request)
    except Exception as e:
        error_msg = f"Error streaming from OpenAI ({model_id_from_request}): {str(e)}"; print(f"ERROR: Stream - {error_msg}")
        yield f"data: {ChatCompletionStreamResponse(id=request_id,model=model_id_from_request,choices=[ChatCompletionStreamChoice(index=0,delta=ChatCompletionStreamChoiceDelta(content=f'Error: {error_msg}'),finish_reason='error')]).model_dump_json()}\n\n"
    finally: yield f"data: [DONE]\n\n"


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
    
    original_user_messages = [ChatMessageInput(role=m.role, content=m.content) for m in request.messages]
    current_user_query_content = ""
    if original_user_messages and original_user_messages[-1].role == "user":
        current_user_query_content = original_user_messages[-1].content

    request_id = f"chatcmpl-{uuid.uuid4()}"

    # --- Handle `!use_project` command ---
    if current_user_query_content.lower().startswith("!use_project "):
        command_parts = current_user_query_content.split(" ", 1)
        response_message = "Invalid command. Usage: !use_project <project_name>"
        if len(command_parts) > 1 and command_parts[1].strip():
            new_project_name = command_parts[1].strip()
            chat_service.add_message_to_database(db, conversation_id_from_owi, "user", current_user_query_content, author_user_id=db_user.id)
            new_project_obj = chat_service.get_or_create_project(db, project_name=new_project_name)
            chat_service.ensure_user_linked_to_project(db, user_id=db_user.id, project_id=new_project_obj.id)
            rag_service.get_or_create_project_collection(project_id=new_project_obj.id) # Ensure Chroma collection exists
            active_project_for_user[db_user.id] = new_project_obj.id
            response_message = f"Switched to project: '{new_project_obj.name}'. RAG and chat are now scoped to this project."
            print(f"INFO:     User '{db_user.id}' switched project to '{new_project_obj.name}' (ID: {new_project_obj.id})")
        
        chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", response_message, author_user_id=None, model_used="system_command")
        if request.stream:
            async def cmd_stream():
                yield f"data: {ChatCompletionStreamResponse(id=request_id, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(content=response_message, role='assistant'))]).model_dump_json()}\n\n"
                yield f"data: {ChatCompletionStreamResponse(id=request_id, model=request.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(), finish_reason='stop')]).model_dump_json()}\n\n"
                yield f"data: [DONE]\n\n"
            return StreamingResponse(cmd_stream(), media_type="text/event-stream")
        else:
            return ChatCompletionResponse(id=request_id, model=request.model, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=response_message), finish_reason="stop")], usage=Usage(prompt_tokens=0, completion_tokens=len(response_message)//4, total_tokens=len(response_message)//4))

    # --- Regular message processing: Save user message, then RAG, then LLM ---
    if current_user_query_content:
        chat_service.add_message_to_database(db, conversation_id_from_owi, "user", current_user_query_content, author_user_id=db_user.id)
    
    messages_for_llm_with_rag_context = list(original_user_messages)
    if current_user_query_content: 
        try:
            relevant_chunks = rag_service.query_project_collection(project_id=current_project_for_turn.id, query_text=current_user_query_content, n_results=settings.RAG_TOP_K)
            if relevant_chunks:
                retrieved_context_str = "\n\n--- Relevant Context Retrieved ---\n" + "\n\n".join(relevant_chunks) + "\n--- End of Context ---\n"
                context_system_message = ChatMessageInput(role="system", content=f"Based on the following context, answer the user's query:\n{retrieved_context_str}")
                # Insert context before the last user message (which is the current query)
                if len(messages_for_llm_with_rag_context) > 0 and messages_for_llm_with_rag_context[-1].role == "user":
                    messages_for_llm_with_rag_context.insert(len(messages_for_llm_with_rag_context) - 1, context_system_message)
                else: # Or just prepend if no clear last user message (should not happen)
                    messages_for_llm_with_rag_context.insert(0, context_system_message)
                print(f"INFO:     RAG: Prepended {len(relevant_chunks)} context chunks for project '{current_project_for_turn.id}'.")
        except Exception as e: print(f"ERROR:    RAG: Failed to retrieve/inject context for project '{current_project_for_turn.id}': {e}")

    # --- Route to LLM ---
    if request.model.startswith("ollama/"):
        if request.stream:
            return StreamingResponse(ollama_chat_stream_generator(db, current_project_for_turn, db_user.id, conversation_id_from_owi, request.model, messages_for_llm_with_rag_context, request_id), media_type="text/event-stream")
        else: # Non-streaming Ollama
            ollama_model_name = request.model.replace("ollama/", "")
            ollama_formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm_with_rag_context]
            payload = {"model": ollama_model_name, "messages": ollama_formatted_messages, "stream": False}
            try:
                response = await http_client.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=120.0)
                response.raise_for_status(); ollama_data = response.json(); final_content = ollama_data.get("message", {}).get("content", "")
                if final_content: chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", final_content, author_user_id=None, model_used=request.model)
                usage_data = Usage(prompt_tokens=ollama_data.get("prompt_eval_count",0), completion_tokens=ollama_data.get("eval_count",0), total_tokens=ollama_data.get("prompt_eval_count",0)+ollama_data.get("eval_count",0))
                return ChatCompletionResponse(id=request_id, model=request.model, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=final_content), finish_reason="stop")], usage=usage_data if usage_data.total_tokens > 0 else None)
            except Exception as e: print(f"ERROR: Non-streaming Ollama: {e}"); raise HTTPException(status_code=500, detail=str(e))

    elif request.model.startswith("gemini/"):
        if request.stream:
            return StreamingResponse(gemini_chat_stream_generator(db, current_project_for_turn, db_user.id, conversation_id_from_owi, request.model, messages_for_llm_with_rag_context, request_id), media_type="text/event-stream")
        else: # Non-streaming Gemini
            # (Simplified: actual non-streaming call and response parsing needed)
            final_content = f"Non-streaming response from {request.model} (placeholder)."
            # 실제 Gemini non-streaming 호출 로직 필요
            # gemini_response = await gemini_sdk_model.generate_content_async(contents=gemini_history_contents, stream=False)
            # final_content = gemini_response.text
            chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", final_content, author_user_id=None, model_used=request.model)
            return ChatCompletionResponse(id=request_id,model=request.model,choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=final_content),finish_reason="stop")],usage=Usage(prompt_tokens=10,completion_tokens=5,total_tokens=15))

    elif request.model.startswith("openai/"):
        if request.stream:
            return StreamingResponse(openai_api_chat_stream_generator(db, current_project_for_turn, db_user.id, conversation_id_from_owi, request.model, messages_for_llm_with_rag_context, request_id), media_type="text/event-stream")
        else: # Non-streaming OpenAI
            # (Simplified: actual non-streaming call and response parsing needed)
            # completion = await openai_client.chat.completions.create(model=actual_openai_model_name, messages=openai_api_messages, stream=False)
            # final_content = completion.choices[0].message.content
            final_content = f"Non-streaming response from {request.model} (placeholder)."
            chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", final_content, author_user_id=None, model_used=request.model)
            return ChatCompletionResponse(id=request_id,model=request.model,choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=final_content),finish_reason="stop")],usage=Usage(prompt_tokens=10,completion_tokens=5,total_tokens=15))
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")

# Ensure OllamaSimplePromptRequest class is defined if you keep /test_ollama_prompt
class OllamaSimplePromptRequest(BaseModel):
    model: str = "gemma:2b"
    prompt: str
    stream: bool = False

@app.post("/test_ollama_prompt", tags=["Ollama Tests"])
async def test_ollama_prompt(request_data: OllamaSimplePromptRequest):
    # ... (keep your existing test_ollama_prompt logic using /api/generate)
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