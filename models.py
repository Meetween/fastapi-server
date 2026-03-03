"""
Pydantic models for Meetween API
"""
from typing import Optional, Literal, Union, List, Dict
from pydantic import BaseModel, Field, field_validator
import base64

taskTypes = Literal[
    "transcription",
    "summary",
    "catch_me_up"
]

"""
Used with jigasi websocket connection
"""
class MeetweenTranscriptionRequest(BaseModel):
    type: Literal["audio"] = "audio"
    participant_id: Optional[str]
    language: Optional[str] = "en-US"
    audio: str = Field(..., description="Base64 encoded audio data")
    meeting_id: str
    room_name: str

    @field_validator('audio')
    @classmethod
    def validate_audio(cls, v: str) -> str:
        """Validate base64 audio data"""
        try:
            decoded = base64.b64decode(v)
            if len(decoded) == 0:
                raise ValueError("Audio data cannot be empty")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 audio data: {e}")

    def get_audio_bytes(self) -> bytes:
        return base64.b64decode(self.audio)

class MeetweenTranscriptionResponse(BaseModel):
    """Response model for transcription endpoints"""
    type: Literal["transcription"] = "transcription"
    meeting_id: str
    participant_id: Optional[str] = None
    transcript: str
    message_id: Optional[str] = None
    timestamp: Optional[float] = None
    is_interim: Optional[bool] = None
    language: Optional[str] = None
    stability: Optional[float] = Field(None, ge=0.0, le=1.0)

class HistoryEntry(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class MeetweenTaskRequest(BaseModel):
    type: taskTypes
    room_name: str
    meeting_id: str
    participant_id: Optional[str]
    language: Optional[str]
    prompt: Optional[str] = Field(None, description="Optional free-form prompt to pass to the LLM", max_length=2048)
    context: Optional[str] = Field(None, description="Optional context string to provide additional information to the model")
    history: Optional[List[HistoryEntry]] = Field(None, description="Optional conversation history.")
    max_new_tokens: Optional[int] = Field(None, ge=256, le=32768, description="Max tokens to generate")
    do_sample: Optional[bool] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature when do_sample=True")
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=2.0, description="Penalty for repeated tokens")

class MeetweenAudioRequest(MeetweenTaskRequest):
    audio: str

    @field_validator('audio')
    @classmethod
    def validate_audio(cls, v: str) -> str:
        try:
            decoded = base64.b64decode(v)
            if len(decoded) == 0:
                raise ValueError("Audio data cannot be empty")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 audio data: {e}")

    def get_audio_bytes(self) -> bytes:
        return base64.b64decode(self.audio)

class MeetweenTextRequest(MeetweenTaskRequest):
    text: str


class ChatRequest(BaseModel):
    """Request for /chat (question and answer) endpoint. History sent via text (e.g. transcription)."""
    user_id: str
    meeting_id: str
    room_name: str
    input_type: Literal["text", "audio"] = "text"
    prompt: str = Field(..., description="User question or prompt")
    context: Optional[str] = Field(None, description="Optional context (e.g. meeting transcription)")
    history: Optional[List[HistoryEntry]] = Field(None, description="Full chat history for multi-turn Q&A")

class MeetweenCatchupRequest(MeetweenTaskRequest):
    max_duration: Optional[float] = None

class MeetweenTaskResponse(BaseModel):
    type: taskTypes
    resultType: Literal["audio", "text"] = "text"
    room_name: str
    meeting_id: str
    result: str
    language: Optional[str]
    participant_id: Optional[str]
    error: Optional[str] = None


class ChatResponse(BaseModel):
    """Response from /chat endpoint - text from TritonClient."""
    result: str
    room_name: str
    meeting_id: str
    user_id: Optional[str] = None


class MeetweenSummarizationResponse(BaseModel):
    """Response model for summarization endpoint"""
    type: Literal["summarization"] = "summarization"
    summary: str
    participant_id: Optional[str] = None
    language: str = "en-US"


class MeetweenErrorResponse(BaseModel):
    """Response model for errors"""
    type: Literal["error"] = "error"
    message: str
    code: Optional[str] = None

LMMRequest = Union[
    MeetweenAudioRequest,
    MeetweenTextRequest,
    MeetweenTaskRequest
]
