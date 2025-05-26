# server/app/openai_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import time
import uuid

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "Gen AI Enable" # Or "ollama", "google", "openai"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

# For Chat Completions Request
class ChatMessageInput(BaseModel): # Renamed to avoid conflict with ChatMessageOutput
    role: str # "system", "user", "assistant"
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessageInput]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0 # OpenAI default
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None # OpenAI default: inf
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None # This is where OWI might send user identifier

# For Streaming Response (Server-Sent Events - SSE)
class ChatCompletionStreamChoiceDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None # Usually role is only in the first chunk for assistant

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatCompletionStreamChoiceDelta
    finish_reason: Optional[str] = None # e.g., "stop", "length"

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str # Model used
    choices: List[ChatCompletionStreamChoice]

# For Non-Streaming Response
class ChatMessageOutput(BaseModel): # Renamed to avoid conflict
    role: str
    content: str

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessageOutput
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None