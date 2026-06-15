"""
Pydantic models for Meetween API
"""
from typing import Optional, Literal, Union, List, Final
from pydantic import BaseModel, ConfigDict, Field, field_validator
from enum import Enum
import base64

class TaskType(str, Enum):
    TEXT_TRANSLATION = "MT"
    SPEECH_TRANSLATION = "ST"
    TEXT_SUMMARY = "TSUM"
    SPEECH_SUMMARY = "SSUM"

taskTypes = Literal[
    "transcription",
    TaskType.TEXT_TRANSLATION,
    TaskType.SPEECH_TRANSLATION,
    TaskType.TEXT_SUMMARY,
    TaskType.SPEECH_SUMMARY,
    "catch_me_up",
    "chat",
    "agendaTracking",
]

agendaStatusTypes = Literal[
    "pending",
    "completed",
    "skipped",
]


class MeetweenTranslationItem(BaseModel):
    target_language: str
    text: str

"""
Used with jigasi websocket connection
"""
class MeetweenWebsocketRequest(BaseModel):
    type: Literal["audio"] = "audio"
    participant_id: Optional[str] = None
    language: Optional[str] = "en-US"
    audio: str = Field(..., description="Base64 encoded audio data")
    meeting_id: str
    room_name: str
    target_languages: Optional[List[str]] = None

    @field_validator('target_languages', mode='before')
    @classmethod
    def normalize_target_languages(cls, v):
        if isinstance(v, str):
            return [v]
        return v

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

class MeetweenWebsocketResponse(MeetweenTranscriptionResponse):
    translations: Optional[List[MeetweenTranslationItem]] = None

class HistoryEntry(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AgendaItem(BaseModel):
    index: int = Field(..., ge=0)
    text: str = Field(..., min_length=1, description="Agenda item text")
    status: agendaStatusTypes = "pending"


class MeetweenTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: taskTypes
    room_name: str
    meeting_id: str
    participant_id: Optional[str] = None
    language: Optional[str] = None
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


class MeetweenCatchupRequest(MeetweenTaskRequest):
    max_duration: Optional[float] = None


class MeetweenAgendaTrackingRequest(BaseModel):
    type: Literal["agendaTracking"] = "agendaTracking"
    meeting_id: str
    room_id: str
    items: List[AgendaItem]
    requested_by: Optional[Literal["moderator", "admin"]] = None


class MeetweenAgendaTrackingResponse(BaseModel):
    taskType: Literal["agendaTracking"] = "agendaTracking"
    meeting_id: str
    room_id: str
    items: List[AgendaItem]

class MeetweenMTRequest(MeetweenTextRequest):
    type: Literal[TaskType.TEXT_TRANSLATION] = TaskType.TEXT_TRANSLATION
    target_language: str

class MeetweenSTRequest(MeetweenAudioRequest):
    type: Literal[TaskType.SPEECH_TRANSLATION] = TaskType.SPEECH_TRANSLATION
    target_language: str

class AgendaTrackingModelOutput(BaseModel):
    """JSON body from the LLM: {\"items\": [AgendaItem, ...]}."""

    model_config = ConfigDict(extra="forbid")
    items: List[AgendaItem]


class MeetweenTaskResponse(BaseModel):
    type: taskTypes
    resultType: Literal["audio", "text"] = "text"
    room_name: str
    meeting_id: str
    result: str
    language: Optional[str] = "en"
    participant_id: Optional[str] = None
    error: Optional[str] = None

class MeetweenTranslationResponse(MeetweenTaskResponse):
    type: Literal[TaskType.TEXT_TRANSLATION, TaskType.SPEECH_TRANSLATION]
    target_language: str

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

