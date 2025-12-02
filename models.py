"""
Pydantic models for Meetween API
"""
from typing import Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator
import base64

taskTypes = Literal[
		"full_transcript",
		"summary",
		"catch_me_up"
]


class MeetweenTranscriptionRequest(BaseModel):
    type: Literal["audio"] = "audio"
    participant_id: Optional[str]
    language: Optional[str] = "en-US"
    audio: str = Field(..., description="Base64 encoded audio data")

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
    participant_id: Optional[str] = None
    transcript: str
    is_final: bool
    confidence: float = Field(ge=0.0, le=1.0)
    transcript_id: Optional[str] = None
    message_id: Optional[int] = None
    timestamp: Optional[float] = None
    is_interim: Optional[bool] = None
    language: Optional[str] = None
    stability: Optional[float] = Field(None, ge=0.0, le=1.0)

class MeetweenTaskRequest(BaseModel):
    type: taskTypes
    participant_id: Optional[str]
    language: str
    room_name: Optional[str]

class MeetweenAudioRequest(MeetweenTaskRequest):
    audio: str


class MeetweenTextRequest(MeetweenTaskRequest):
    text: str

class MeetweenTaskResponse(BaseModel):
    type: taskTypes
    resultType: Literal["audio", "text"] = "text"
    participant_id: Optional[str]
    language: str
    room_name: str
    result: str
    error: Optional[str] = None


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
    MeetweenTaskRequest,
    MeetweenAudioRequest,
    MeetweenTextRequest
]
