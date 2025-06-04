# server/app/core/llm_services.py
import httpx
import json
import uuid
import time
from typing import List, Dict, Optional, AsyncGenerator, Union 

import google.generativeai as genai
from openai import AsyncOpenAI
from sqlalchemy.orm import Session
import logging

from ..config import settings
from ..openai_models import (
    ChatMessageInput, ChatCompletionStreamResponse, ChatCompletionStreamChoice, 
    ChatCompletionStreamChoiceDelta, ChatCompletionResponse, ChatCompletionChoice, 
    ChatMessageOutput, Usage
)
from ..services import chat_service # For saving assistant messages
from ..db import models as db_models # For type hinting project_obj

logger = logging.getLogger(__name__)

# --- Global Clients ---
http_client = httpx.AsyncClient() # For Ollama
openai_llm_client: Optional[AsyncOpenAI] = None

# --- Client Initialization & Configuration ---
def initialize_llm_clients():
    """Called from main.py startup to initialize clients."""
    global openai_llm_client, http_client

    if http_client.is_closed:
        http_client = httpx.AsyncClient()
        logger.info("LLMService: Re-initialized httpx.AsyncClient for Ollama.")

    if settings.OPENAI_API_KEY:
        try:
            openai_llm_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("LLMService: OpenAI client initialized.")
        except Exception as e:
            openai_llm_client = None 
            logger.error(f"LLMService: Failed to initialize OpenAI client: {e}", exc_info=True)
    else:
        logger.info("LLMService: OpenAI API key not found in settings. OpenAI models will be unavailable.")

    if settings.GEMINI_API_KEY:
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            logger.info("LLMService: Gemini client configured.")
        except Exception as e:
            logger.error(f"LLMService: Failed to configure Gemini client: {e}", exc_info=True)
    else:
        logger.info("LLMService: Gemini API key not found in settings. Gemini models will be unavailable.")

async def close_llm_clients():
    """Called from main.py shutdown."""
    global openai_llm_client, http_client
    if not http_client.is_closed:
        await http_client.aclose()
        logger.info("LLMService: httpx client (for Ollama) closed.")
    if openai_llm_client:
        await openai_llm_client.close()
        logger.info("LLMService: OpenAI client closed.")
    logger.info("LLMService: All LLM related clients closed.")

async def _yield_error_sse(request_id: str, model_id: str, error_msg: str) -> AsyncGenerator[str, None]:
    logger.error(f"LLMService ({model_id}): {error_msg}")
    error_chunk_obj = ChatCompletionStreamResponse(
        id=request_id, model=model_id,
        choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(content=f"Error: {error_msg}"), finish_reason="error")]
    )
    yield f"data: {error_chunk_obj.model_dump_json()}\n\n"

async def _pull_ollama_model_if_missing(model_name_with_tag: str) -> bool:
    if not settings.OLLAMA_BASE_URL: 
        logger.error(f"Ollama base URL not configured. Cannot pull model {model_name_with_tag}.")
        return False
    try:
        response = await http_client.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/show", json={"name": model_name_with_tag}, timeout=30.0)
        if response.status_code == 200:
            logger.info(f"Ollama model '{model_name_with_tag}' already exists locally.")
            return True
    except Exception as e:
        logger.warning(f"Could not verify existence of Ollama model '{model_name_with_tag}': {e}. Attempting pull.")

    logger.info(f"Attempting to pull Ollama model: {model_name_with_tag}...")
    pull_payload = {"name": model_name_with_tag, "stream": True} 
    pull_success = False
    try:
        async with http_client.stream("POST", f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/pull", json=pull_payload, timeout=3600.0) as response:
            if response.status_code != 200:
                 content_bytes = await response.aread()
                 logger.error(f"Failed to start pull for Ollama model '{model_name_with_tag}'. Status: {response.status_code}, Response: {content_bytes.decode(errors='ignore')}")
                 return False
            
            last_log_progress = -10 
            async for line in response.aiter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        status = chunk.get("status", "")
                        if "total" in chunk and "completed" in chunk and chunk["total"] > 0:
                            progress = (chunk['completed'] / chunk['total']) * 100
                            if progress == 0 or progress >= last_log_progress + 10 or progress == 100:
                                logger.info(f"Pulling {model_name_with_tag}: {status} - {progress:.1f}%")
                                last_log_progress = progress
                        elif status: 
                            logger.info(f"Pulling {model_name_with_tag}: {status}")
                        if "error" in chunk:
                            logger.error(f"Error pulling Ollama model '{model_name_with_tag}': {chunk['error']}")
                            return False
                        if status == "success": pull_success = True
                    except json.JSONDecodeError: logger.warning(f"Non-JSON response while pulling {model_name_with_tag}: {line}")
            
            if response.status_code == 200 and pull_success:
                 logger.info(f"Ollama model '{model_name_with_tag}' pulled successfully.")
                 return True
            elif not pull_success and response.status_code == 200 : 
                 logger.info(f"Ollama model pull for '{model_name_with_tag}' stream ended with HTTP 200. Assuming success.")
                 return True
            logger.error(f"Failed to pull Ollama model '{model_name_with_tag}'. Final status: {response.status_code}")
            return False
    except httpx.TimeoutException: logger.error(f"Timeout pulling Ollama model '{model_name_with_tag}'."); return False
    except Exception as e: logger.error(f"Error pulling Ollama model '{model_name_with_tag}': {e}", exc_info=True); return False

# CORRECTED FUNCTION NAME HERE
async def ensure_default_ollama_models_are_pulled_async():
    if not settings.DEFAULT_OLLAMA_MODELS_TO_PULL:
        logger.info("No default Ollama models specified to pull.")
        return
    logger.info(f"Starting background check/pull for default Ollama models: {settings.DEFAULT_OLLAMA_MODELS_TO_PULL}")
    for model_name in settings.DEFAULT_OLLAMA_MODELS_TO_PULL:
        await _pull_ollama_model_if_missing(model_name)
    logger.info("Background check/pull for default Ollama models finished.")

# --- Ollama Stream Generator ---
async def _ollama_chat_stream_generator(db: Session, project_obj: db_models.Project, user_identifier: str, conversation_id_from_owi: str, model_id: str, messages_for_llm: List[ChatMessageInput], request_id: str) -> AsyncGenerator[str, None]:
    # ... (content as before)
    ollama_model_name = model_id.replace("ollama/", "")
    ollama_formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm]
    payload = {"model": ollama_model_name, "messages": ollama_formatted_messages, "stream": True}
    assistant_response_buffer = ""
    response_obj_for_status_check = None; error_occurred = False
    try:
        async with http_client.stream("POST", f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=120.0) as response:
            response_obj_for_status_check = response
            if response.status_code != 200:
                error_content = await response.aread(); error_msg = f"Ollama API Error ({response.status_code}): {error_content.decode()}"
                async for item in _yield_error_sse(request_id, model_id, error_msg): yield item
                error_occurred = True 
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
                        except json.JSONDecodeError: logger.warning(f"LLMService (Ollama): Could not decode JSON line: {line}")
        if not error_occurred and assistant_response_buffer and (getattr(response_obj_for_status_check, 'status_code', 500) == 200):
            chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", assistant_response_buffer, author_user_id=None, model_used=model_id)
    except httpx.RequestError as e:
        async for item in _yield_error_sse(request_id, model_id, f"Ollama request error: {e}"):
            yield item
    except Exception as e:
        async for item in _yield_error_sse(request_id, model_id, f"Ollama stream error: {e}"):
            yield item
    finally:
        yield f"data: [DONE]\n\n"

# --- Gemini Stream Generator ---
async def _gemini_chat_stream_generator(db: Session, project_obj: db_models.Project, user_identifier: str, conversation_id_from_owi: str, model_id_from_request: str, messages_for_llm: List[ChatMessageInput], request_id: str) -> AsyncGenerator[str, None]:
    # ... (content as before)
    if not settings.GEMINI_API_KEY:
        async for item in _yield_error_sse(request_id, model_id_from_request, "Gemini API key not configured."): yield item
        yield f"data: [DONE]\n\n"; return
    assistant_response_buffer = ""
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
        response_stream = await gemini_sdk_model.generate_content_async(contents=gemini_history_contents, stream=True)
        first_chunk = True; final_finish_reason_from_sdk = None
        async for chunk in response_stream:
            delta_content = ""
            if hasattr(chunk, 'text') and chunk.text: delta_content = chunk.text
            elif hasattr(chunk, 'parts') and chunk.parts: 
                for part in chunk.parts: 
                    if hasattr(part, 'text'): delta_content += part.text
            if hasattr(chunk, 'candidates') and chunk.candidates and hasattr(chunk.candidates[0], 'finish_reason') and chunk.candidates[0].finish_reason:
                reason_val = str(chunk.candidates[0].finish_reason).lower()
                if "stop" in reason_val: final_finish_reason_from_sdk = "stop"
                elif "length" in reason_val: final_finish_reason_from_sdk = "length"
            if delta_content: assistant_response_buffer += delta_content
            delta_role_to_send = "assistant" if first_chunk and delta_content else None
            if delta_content: first_chunk = False
            yield f"data: {ChatCompletionStreamResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(content=delta_content if delta_content else None, role=delta_role_to_send), finish_reason=final_finish_reason_from_sdk)]).model_dump_json()}\n\n"
            if final_finish_reason_from_sdk: break 
        if not final_finish_reason_from_sdk: 
            yield f"data: {ChatCompletionStreamResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(), finish_reason='stop')]).model_dump_json()}\n\n"
        if assistant_response_buffer:
            chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", assistant_response_buffer, author_user_id=None, model_used=model_id_from_request)
    except Exception as e:
        async for item in _yield_error_sse(request_id, model_id_from_request, f"Gemini stream error: {e}"):
            yield item
    finally: yield f"data: [DONE]\n\n"

# --- OpenAI Stream Generator ---
async def _openai_api_chat_stream_generator(db: Session, project_obj: db_models.Project, user_identifier: str, conversation_id_from_owi: str, model_id_from_request: str, messages_for_llm: List[ChatMessageInput], request_id: str) -> AsyncGenerator[str, None]:
    # ... (content as before)
    global openai_llm_client
    if not openai_llm_client:
        async for item in _yield_error_sse(request_id, model_id_from_request, "OpenAI client not initialized."): yield item
        yield f"data: [DONE]\n\n"; return
    assistant_response_buffer = ""
    try:
        actual_openai_model_name = model_id_from_request.split('/', 1)[1]
        openai_api_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm]
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
async def route_chat_to_llm_stream(db: Session, project_obj: db_models.Project, user_identifier: str, conversation_id_from_owi: str, model_id_from_request: str, messages_for_llm: List[ChatMessageInput], request_id: str) -> AsyncGenerator[str, None]:
    # ... (content as before)
    logger.debug(f"Routing stream request for model '{model_id_from_request}' in project '{project_obj.id}' by user '{user_identifier}'")
    if model_id_from_request.startswith("ollama/"):
        async for chunk in _ollama_chat_stream_generator(db, project_obj, user_identifier, conversation_id_from_owi, model_id_from_request, messages_for_llm, request_id): yield chunk
    elif model_id_from_request.startswith("gemini/"):
        async for chunk in _gemini_chat_stream_generator(db, project_obj, user_identifier, conversation_id_from_owi, model_id_from_request, messages_for_llm, request_id): yield chunk
    elif model_id_from_request.startswith("openai/"):
        async for chunk in _openai_api_chat_stream_generator(db, project_obj, user_identifier, conversation_id_from_owi, model_id_from_request, messages_for_llm, request_id): yield chunk
    else: 
        async for item in _yield_error_sse(request_id, model_id_from_request, f"Unsupported model for streaming: {model_id_from_request}"): yield item
        yield f"data: [DONE]\n\n"

# --- Main Dispatcher for LLM Chat (Non-Streaming) ---
async def route_chat_to_llm_non_stream(db: Session, project_obj: db_models.Project, user_identifier: str, conversation_id_from_owi: str, model_id_from_request: str, messages_for_llm: List[ChatMessageInput], request_id: str) -> ChatCompletionResponse:
    # ... (content as before, ensure logging is used)
    logger.debug(f"Routing non-stream request for model '{model_id_from_request}' in project '{project_obj.id}' by user '{user_identifier}'")
    if model_id_from_request.startswith("ollama/"):
        ollama_model_name = model_id_from_request.replace("ollama/", "")
        ollama_formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm]
        payload = {"model": ollama_model_name, "messages": ollama_formatted_messages, "stream": False}
        try:
            response = await http_client.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=120.0)
            response.raise_for_status()
            ollama_data = response.json()
            final_content = ollama_data.get("message", {}).get("content", "")
            if final_content: chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", final_content, author_user_id=None, model_used=model_id_from_request)
            usage_data = Usage(prompt_tokens=ollama_data.get("prompt_eval_count",0), completion_tokens=ollama_data.get("eval_count",0), total_tokens=ollama_data.get("prompt_eval_count",0)+ollama_data.get("eval_count",0))
            return ChatCompletionResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=final_content or ""), finish_reason="stop")], usage=usage_data if usage_data.total_tokens > 0 else None)
        except Exception as e: logger.error(f"LLMService (Ollama non-stream): {e}", exc_info=True); return ChatCompletionResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=f"Error contacting Ollama: {e}"), finish_reason="error")])
    elif model_id_from_request.startswith("gemini/"):
        if not settings.GEMINI_API_KEY: return ChatCompletionResponse(id=request_id,model=model_id_from_request,choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content="Error: Gemini API key not configured"),finish_reason="error")])
        try:
            actual_gemini_model_name = model_id_from_request.split('/', 1)[1]; 
            if actual_gemini_model_name not in ["gemini-pro", "gemini-1.0-pro", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"] and not actual_gemini_model_name.startswith("models/"): actual_gemini_model_name = f"models/{actual_gemini_model_name}"
            gemini_history_contents = []; system_instruction = None; temp_messages = list(messages_for_llm)
            if temp_messages and temp_messages[0].role == "system": system_instruction = temp_messages.pop(0).content
            for msg in temp_messages: gemini_history_contents.append({'role': "model" if msg.role == "assistant" else msg.role, 'parts': [{'text': msg.content}]})
            model_init_args = {}; 
            if system_instruction: model_init_args['system_instruction'] = system_instruction
            gemini_sdk_model = genai.GenerativeModel(actual_gemini_model_name, **model_init_args)
            gemini_response = await gemini_sdk_model.generate_content_async(contents=gemini_history_contents, stream=False)
            final_content = "";
            if hasattr(gemini_response, 'text') and gemini_response.text: final_content = gemini_response.text
            elif hasattr(gemini_response, 'parts') and gemini_response.parts:
                 for part in gemini_response.parts: 
                    if hasattr(part, 'text'): final_content += part.text
            elif hasattr(gemini_response, 'candidates') and gemini_response.candidates:
                for candidate in gemini_response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'): final_content += part.text
            prompt_tokens = len(json.dumps(gemini_history_contents)) // 4 ; completion_tokens = len(final_content) // 4
            if final_content: chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", final_content, author_user_id=None, model_used=model_id_from_request)
            return ChatCompletionResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=final_content), finish_reason="stop")], usage=Usage(prompt_tokens=prompt_tokens,completion_tokens=completion_tokens,total_tokens=prompt_tokens+completion_tokens))
        except Exception as e: logger.error(f"LLMService (Gemini non-stream): {e}", exc_info=True); return ChatCompletionResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=f"Error contacting Gemini: {e}"), finish_reason="error")])
    elif model_id_from_request.startswith("openai/"):
        global openai_llm_client
        if not openai_llm_client: return ChatCompletionResponse(id=request_id,model=model_id_from_request,choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content="Error: OpenAI client not initialized"),finish_reason="error")])
        try:
            actual_openai_model_name = model_id_from_request.split('/', 1)[1]
            openai_api_messages = [{"role": msg.role, "content": msg.content} for msg in messages_for_llm]
            completion = await openai_llm_client.chat.completions.create(model=actual_openai_model_name, messages=openai_api_messages, stream=False)
            final_content = completion.choices[0].message.content
            usage_data = Usage(prompt_tokens=completion.usage.prompt_tokens, completion_tokens=completion.usage.completion_tokens, total_tokens=completion.usage.total_tokens)
            if final_content: chat_service.add_message_to_database(db, conversation_id_from_owi, "assistant", final_content or "", author_user_id=None, model_used=model_id_from_request)
            return ChatCompletionResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=final_content or ""), finish_reason=completion.choices[0].finish_reason)], usage=usage_data)
        except Exception as e: logger.error(f"LLMService (OpenAI non-stream): {e}", exc_info=True); return ChatCompletionResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=f"Error contacting OpenAI: {e}"), finish_reason="error")])
    else:
        logger.error(f"LLMService (Non-stream): Unsupported model '{model_id_from_request}'")
        return ChatCompletionResponse(id=request_id, model=model_id_from_request, choices=[ChatCompletionChoice(index=0, message=ChatMessageOutput(role="assistant", content=f"Error: Unsupported model '{model_id_from_request}'"), finish_reason="error")])

