# server/app/core/llm_services.py
import httpx
import json
import uuid
import time
from typing import List, Dict, Optional, AsyncGenerator
import logging # Added

import google.generativeai as genai
from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from ..openai_models import (
    ChatMessageInput, ChatCompletionStreamResponse, ChatCompletionStreamChoice, 
    ChatCompletionStreamChoiceDelta, ChatCompletionResponse, ChatCompletionChoice, 
    ChatMessageOutput, Usage # Make sure all needed models are here
)

from ..config import settings
from ..services import chat_service # For saving assistant messages
from ..db import models as db_models # For type hinting project_obj

logger = logging.getLogger(__name__) # Added

# --- Global Clients (initialized by main.py's startup event) ---
# Re-usable HTTP client primarily for Ollama
http_client = httpx.AsyncClient()
# OpenAI client
openai_llm_client: Optional[AsyncOpenAI] = None

# --- Client Initialization & Configuration ---
def initialize_llm_clients():
    """Called from main.py startup to initialize clients."""
    global openai_llm_client
    if settings.OPENAI_API_KEY:
        try:
            openai_llm_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("LLMService: OpenAI client initialized.")
        except Exception as e:
            logger.error(f"LLMService: Failed to initialize OpenAI client: {e}")
            openai_llm_client = None # Ensure it's None if init fails
    else:
        logger.info("LLMService: OpenAI API key not found. OpenAI models will be unavailable.")

    if settings.GEMINI_API_KEY:
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            logger.info("LLMService: Gemini client configured.")
        except Exception as e:
            logger.error(f"LLMService: Failed to configure Gemini client: {e}")
    else:
        logger.info("LLMService: Gemini API key not found. Gemini models will be unavailable.")

async def pull_ollama_model(model_name: str):
    """Pulls a specific model into Ollama if it doesn't exist."""
    if not settings.OLLAMA_BASE_URL:
        logger.error(f"Ollama base URL not configured. Cannot pull model {model_name}.")
        return False
    
    # Check if model exists first (optional, as /api/pull might handle it)
    try:
        tags_response = await http_client.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags")
        tags_response.raise_for_status()
        available_models = tags_response.json().get("models", [])
        for model_info in available_models:
            if model_info.get("name") == model_name:
                logger.info(f"Ollama model '{model_name}' already exists locally.")
                return True
    except Exception as e:
        logger.warning(f"Could not check existing Ollama models before pulling '{model_name}': {e}. Proceeding with pull attempt.")

    logger.info(f"Attempting to pull Ollama model: {model_name}...")
    pull_payload = {"name": model_name, "stream": False} # Stream false for a single status update
    try:
        # Ollama's pull can be long-running. Increase timeout significantly.
        # The stream=False will make it block until done or error.
        # For a better UX during startup, stream=True and handling progress would be ideal,
        # but for v1, a blocking pull with a long timeout on startup is simpler to implement.
        async with http_client.stream("POST", f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/pull", json=pull_payload, timeout=1800.0) as response: # 30 min timeout
            async for line in response.aiter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        status = chunk.get("status", "")
                        # Basic progress logging
                        if "total" in chunk and "completed" in chunk:
                            progress = (chunk['completed'] / chunk['total']) * 100
                            logger.info(f"Pulling {model_name}: {status} - {progress:.2f}%")
                        else:
                            logger.info(f"Pulling {model_name}: {status}")
                        
                        if chunk.get("error"):
                            logger.error(f"Error pulling Ollama model '{model_name}': {chunk.get('error')}")
                            return False
                        # "success" status might appear in the last chunk if stream=false wasn't fully respected by client or if API changed
                        # For stream=false, it should ideally be a single JSON response after completion.
                        # However, robustly checking last status or overall response status code is better.
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON response while pulling {model_name}: {line}")
            
            if response.status_code == 200: # Check HTTP status after stream
                 logger.info(f"Ollama model '{model_name}' pulled successfully (or was already up to date).")
                 return True
            else:
                error_content = await response.aread()
                logger.error(f"Failed to pull Ollama model '{model_name}'. Status: {response.status_code}, Response: {error_content.decode()}")
                return False

    except httpx.TimeoutException:
        logger.error(f"Timeout while trying to pull Ollama model '{model_name}'. This can happen with large models.")
        return False
    except Exception as e:
        logger.error(f"An error occurred while pulling Ollama model '{model_name}': {e}", exc_info=True)
        return False

async def ensure_default_ollama_models():
    """Checks and pulls default Ollama models specified in settings."""
    if not settings.DEFAULT_OLLAMA_MODELS_TO_PULL:
        logger.info("No default Ollama models specified in settings to pull.")
        return

    logger.info(f"Checking/Pulling default Ollama models: {settings.DEFAULT_OLLAMA_MODELS_TO_PULL}")
    for model_name in settings.DEFAULT_OLLAMA_MODELS_TO_PULL:
        await pull_ollama_model(model_name) # Pull one by one for clearer logging

async def close_llm_clients():
    """Called from main.py shutdown."""
    global openai_llm_client
    if not http_client.is_closed:
        await http_client.aclose()
    if openai_llm_client:
        await openai_llm_client.close()
    logger.info("LLMService: HTTP clients (Ollama, OpenAI) closed.")

# --- Helper to yield error stream consistently ---
async def _yield_error_sse(request_id: str, model_id: str, error_msg: str) -> AsyncGenerator[str, None]:
    logger.error(f"LLMService ({model_id}): {error_msg}")
    error_chunk = ChatCompletionStreamResponse(
        id=request_id, model=model_id,
        choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(content=f"Error: {error_msg}"), finish_reason="error")]
    )
    yield f"data: {error_chunk.model_dump_json()}\n\n"
    yield f"data: [DONE]\n\n"

# --- Ollama Stream Generator ---
async def ollama_chat_stream_generator(
    db: Session, project_obj: db_models.Project, user_identifier: str, 
    conversation_id_from_owi: str, model_id: str, 
    messages_for_llm: List[ChatMessageInput], request_id: str
) -> AsyncGenerator[str, None]:
    ollama_model_name = model_id.replace("ollama/", "")
    ollama_formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm]
    payload = {"model": ollama_model_name, "messages": ollama_formatted_messages, "stream": True}
    assistant_response_buffer = ""
    try:
        async with http_client.stream("POST", f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=120.0) as response:
            if response.status_code != 200:
                error_content = await response.aread(); error_msg = f"Ollama API Error ({response.status_code}): {error_content.decode()}"
                async for item in _yield_error_sse(request_id, model_id, error_msg): yield item; return
            
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
                    except json.JSONDecodeError: 
                        logger.warning(f"LLMService (Ollama): Could not decode JSON line: {line}")
        
        if assistant_response_buffer and (getattr(response, 'status_code', 500) == 200):
            chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", assistant_response_buffer, author_user_id=None, model_used=model_id)
    except httpx.RequestError as e: 
        async for item in _yield_error_sse(request_id, model_id, f"Request error (Ollama stream): {e}"): # This calls _yield_error_sse
            yield item 
    except Exception as e: 
        async for item in _yield_error_sse(request_id, model_id, f"Generic error (Ollama stream): {e}"): # This calls _yield_error_sse
            yield item 
    finally: 
        yield f"data: [DONE]\n\n"


# --- Gemini Stream Generator ---
async def gemini_chat_stream_generator(
    db: Session, project_obj: db_models.Project, user_identifier: str, 
    conversation_id_from_owi: str, model_id_from_request: str, 
    messages_for_llm: List[ChatMessageInput], request_id: str
) -> AsyncGenerator[str, None]:
    if not settings.GEMINI_API_KEY:
        async for item in _yield_error_sse(request_id, model_id_from_request, "Gemini API key not configured."): yield item; return
    try:
        actual_gemini_model_name = model_id_from_request.split('/', 1)[1]
        # Add "models/" prefix if it's a specific version like "gemini-1.5-flash-latest" but not for "gemini-pro"
        if "pro" not in actual_gemini_model_name and not actual_gemini_model_name.startswith("models/"):
             actual_gemini_model_name = f"models/{actual_gemini_model_name}"


        gemini_history_contents = []; system_instruction = None; temp_messages = list(messages_for_llm)
        if temp_messages and temp_messages[0].role == "system": system_instruction = temp_messages.pop(0).content
        for msg in temp_messages: gemini_history_contents.append({'role': "model" if msg.role == "assistant" else msg.role, 'parts': [{'text': msg.content}]})
        
        model_init_args = {}
        if system_instruction: model_init_args['system_instruction'] = system_instruction
        gemini_sdk_model = genai.GenerativeModel(actual_gemini_model_name, **model_init_args)
        
        response_stream = await gemini_sdk_model.generate_content_async(contents=gemini_history_contents, stream=True)
        assistant_response_buffer = ""; first_chunk = True;
        async for chunk in response_stream:
            delta_content = ""; finish_reason_from_chunk = None
            if hasattr(chunk, 'text') and chunk.text: delta_content = chunk.text
            elif hasattr(chunk, 'parts') and chunk.parts: 
                for part in chunk.parts: 
                    if hasattr(part, 'text'): delta_content += part.text
            
            # Check for finish reason from Gemini chunk (this can be complex as it's in candidate)
            # if chunk.candidates and chunk.candidates[0].finish_reason:
            #     finish_reason_val = str(chunk.candidates[0].finish_reason).lower()
            #     if "stop" in finish_reason_val: finish_reason_from_chunk = "stop"
            #     elif "length" in finish_reason_val: finish_reason_from_chunk = "length"
            #     # other reasons: "SAFETY", "RECITATION", "OTHER"
            
            if delta_content: assistant_response_buffer += delta_content
            delta_role_to_send = "assistant" if first_chunk and delta_content else None
            if delta_content: first_chunk = False

            stream_choice = ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(content=delta_content if delta_content else None, role=delta_role_to_send), finish_reason=finish_reason_from_chunk)
            yield f"data: {ChatCompletionStreamResponse(id=request_id, model=model_id_from_request, choices=[stream_choice]).model_dump_json()}\n\n"
        
        # If loop finishes and no explicit stop from Gemini, send one
        if not finish_reason_from_chunk :
             yield f"data: {ChatCompletionStreamResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(), finish_reason='stop')]).model_dump_json()}\n\n"

        if assistant_response_buffer:
            chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", assistant_response_buffer, author_user_id=None, model_used=model_id_from_request)
    except Exception as e:
        async for item in _yield_error_sse(request_id, model_id_from_request, f"Gemini stream error: {e}"):
            yield item
    finally:
        yield f"data: [DONE]\n\n"


# --- OpenAI Stream Generator ---
async def openai_api_chat_stream_generator(
    db: Session, project_obj: db_models.Project, user_identifier: str, 
    conversation_id_from_owi: str, model_id_from_request: str, 
    messages_for_llm: List[ChatMessageInput], request_id: str
) -> AsyncGenerator[str, None]:
    global openai_llm_client
    if not openai_llm_client: # Check if client was initialized
        async for item in _yield_error_sse(request_id, model_id_from_request, "OpenAI client not initialized (check API key)."): yield item; return
    try:
        actual_openai_model_name = model_id_from_request.split('/', 1)[1]
        openai_api_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm]
        assistant_response_buffer = ""
        stream = await openai_llm_client.chat.completions.create(model=actual_openai_model_name, messages=openai_api_messages, stream=True)
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
        async for item in _yield_error_sse(request_id, model_id_from_request, f"OpenAI stream error: {e}"):
            yield item
    finally: yield f"data: [DONE]\n\n"


# --- Main Dispatcher for LLM Chat (Streaming) ---
async def route_chat_to_llm_stream(
    db: Session, 
    project_obj: db_models.Project, 
    user_identifier: str, 
    conversation_id_from_owi: str, 
    model_id_from_request: str, 
    messages_for_llm: List[ChatMessageInput], 
    request_id: str
) -> AsyncGenerator[str, None]:
    if model_id_from_request.startswith("ollama/"):
        async for chunk in ollama_chat_stream_generator(db, project_obj, user_identifier, conversation_id_from_owi, model_id_from_request, messages_for_llm, request_id):
            yield chunk
    elif model_id_from_request.startswith("gemini/"):
        async for chunk in gemini_chat_stream_generator(db, project_obj, user_identifier, conversation_id_from_owi, model_id_from_request, messages_for_llm, request_id):
            yield chunk
    elif model_id_from_request.startswith("openai/"):
        async for chunk in openai_api_chat_stream_generator(db, project_obj, user_identifier, conversation_id_from_owi, model_id_from_request, messages_for_llm, request_id):
            yield chunk
    else:
        async for item in _yield_error_sse(request_id, model_id_from_request, f"Unsupported model for streaming: {model_id_from_request}"):
            yield item

# TODO: Implement route_chat_to_llm_non_stream for non-streaming responses
async def route_chat_to_llm_non_stream(
    db: Session, 
    project_obj: db_models.Project, 
    user_identifier: str, 
    conversation_id_from_owi: str, 
    model_id_from_request: str, 
    messages_for_llm: List[ChatMessageInput], 
    request_id: str
) -> ChatCompletionResponse: # This should return the full ChatCompletionResponse model
    # --- Ollama Non-Streaming ---
    if model_id_from_request.startswith("ollama/"):
        ollama_model_name = model_id_from_request.replace("ollama/", "")
        ollama_formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm]
        payload = {"model": ollama_model_name, "messages": ollama_formatted_messages, "stream": False}
        try:
            response = await http_client.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=120.0)
            response.raise_for_status()
            ollama_data = response.json()
            final_content = ollama_data.get("message", {}).get("content", "")
            if final_content:
                chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", final_content, author_user_id=None, model_used=model_id_from_request)
            
            usage_data = Usage(
                prompt_tokens=ollama_data.get("prompt_eval_count",0), 
                completion_tokens=ollama_data.get("eval_count",0), 
                total_tokens=ollama_data.get("prompt_eval_count",0)+ollama_data.get("eval_count",0)
            )
            return ChatCompletionResponse(
                id=request_id, model=model_id_from_request,
                choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=final_content), finish_reason="stop")],
                usage=usage_data if usage_data.total_tokens > 0 else None
            )
        except Exception as e:
            logger.error(f"LLMService (Ollama non-stream): {e}")
            # Fallback to returning an error in the ChatCompletionResponse structure
            return ChatCompletionResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=f"Error: {e}"), finish_reason="error")])

    # --- Gemini Non-Streaming (Placeholder/Simplified) ---
    elif model_id_from_request.startswith("gemini/"):
        if not settings.GEMINI_API_KEY:
            return ChatCompletionResponse(id=request_id,model=model_id_from_request,choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content="Error: Gemini API key not configured"),finish_reason="error")])
        try:
            actual_gemini_model_name = model_id_from_request.split('/', 1)[1]
            if actual_gemini_model_name not in ["gemini-pro", "gemini-1.0-pro", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"] and not actual_gemini_model_name.startswith("models/"):
                actual_gemini_model_name = f"models/{actual_gemini_model_name}"

            gemini_history_contents = []; system_instruction = None; temp_messages = list(messages_for_llm)
            if temp_messages and temp_messages[0].role == "system": system_instruction = temp_messages.pop(0).content
            for msg in temp_messages: gemini_history_contents.append({'role': "model" if msg.role == "assistant" else msg.role, 'parts': [{'text': msg.content}]})
            model_init_args = {}; 
            if system_instruction: model_init_args['system_instruction'] = system_instruction
            gemini_sdk_model = genai.GenerativeModel(actual_gemini_model_name, **model_init_args)

            gemini_response = await gemini_sdk_model.generate_content_async(contents=gemini_history_contents, stream=False)
            final_content = ""
            if hasattr(gemini_response, 'text') and gemini_response.text: final_content = gemini_response.text
            elif hasattr(gemini_response, 'parts') and gemini_response.parts:
                 for part in gemini_response.parts: 
                    if hasattr(part, 'text'): final_content += part.text
            # TODO: Parse usage from gemini_response if available

            if final_content:
                chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", final_content, author_user_id=None, model_used=model_id_from_request)
            return ChatCompletionResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=final_content), finish_reason="stop")]) # Dummy usage
        except Exception as e:
            logger.error(f"LLMService (Gemini non-stream): {e}")
            return ChatCompletionResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=f"Error: {e}"), finish_reason="error")])


    # --- OpenAI Non-Streaming (Placeholder/Simplified) ---
    elif model_id_from_request.startswith("openai/"):
        global openai_llm_client
        if not openai_llm_client:
            return ChatCompletionResponse(id=request_id,model=model_id_from_request,choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content="Error: OpenAI client not initialized"),finish_reason="error")])
        try:
            actual_openai_model_name = model_id_from_request.split('/', 1)[1]
            openai_api_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm]
            
            completion = await openai_llm_client.chat.completions.create(model=actual_openai_model_name, messages=openai_api_messages, stream=False)
            final_content = completion.choices[0].message.content
            usage_data = Usage(prompt_tokens=completion.usage.prompt_tokens, completion_tokens=completion.usage.completion_tokens, total_tokens=completion.usage.total_tokens)

            if final_content:
                chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", final_content, author_user_id=None, model_used=model_id_from_request)
            return ChatCompletionResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=final_content or ""), finish_reason=completion.choices[0].finish_reason)], usage=usage_data)
        except Exception as e:
            logger.error(f"LLMService (OpenAI non-stream): {e}")
            return ChatCompletionResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=f"Error: {e}"), finish_reason="error")])

    else:
        return ChatCompletionResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=f"Error: Unsupported model '{model_id_from_request}'"), finish_reason="error")])