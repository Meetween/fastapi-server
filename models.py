"""
Pydantic models for Meetween API
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field
import uuid


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


class TranscriptionResult(BaseModel):
    """Internal transcription result model"""
    text: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.95)
    language: str = "en-US"
    error: Optional[str] = None

    @property
    def has_error(self) -> bool:
        return self.error is not None
