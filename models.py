"""
Pydantic models for Whisper and Meetween protocols
"""
from typing import Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator
import base64
import uuid


# ===== WHISPER PROTOCOL MODELS =====

class WhisperHeader(BaseModel):
    """
    Whisper protocol header parsed from 60-byte binary header
    Format: "participant_id|language" (padded with null bytes)
    """
    participant_id: str = Field(..., description="Participant identifier")
    language: str = Field(default="en", description="Language code (BCP-47)")

    @classmethod
    def from_bytes(cls, header_bytes: bytes) -> "WhisperHeader":
        """Parse Whisper header from 60-byte binary data"""
        if len(header_bytes) != 60:
            raise ValueError(f"Header must be exactly 60 bytes, got {len(header_bytes)}")

        header_str = header_bytes.decode('utf-8', errors='ignore').rstrip('\x00')

        try:
            participant_id, language = header_str.split('|', 1)
        except ValueError:
            raise ValueError(f"Invalid header format: '{header_str}'")

        return cls(participant_id=participant_id.strip(), language=language.strip())


class WhisperAudioMessage(BaseModel):
    """Complete Whisper audio message with parsed header and audio data"""
    participant_id: str
    language: str
    audio_data: bytes = Field(..., repr=False)

    @classmethod
    def from_binary(cls, data: bytes) -> "WhisperAudioMessage":
        """Parse complete Whisper message from binary data"""
        if len(data) <= 60:
            raise ValueError(f"Message too short: {len(data)} bytes")

        header = WhisperHeader.from_bytes(data[:60])
        audio_data = data[60:]

        return cls(
            participant_id=header.participant_id,
            language=header.language,
            audio_data=audio_data
        )


class WhisperResponse(BaseModel):
    """Response sent back to Whisper client"""
    type: Literal["interim", "final"]
    participant_id: str
    text: str
    variance: float = Field(ge=0.0, le=1.0)


# ===== MEETWEEN PROTOCOL MODELS =====

class MeetweenAudioRequest(BaseModel):
    type: Literal["audio"] = "audio"
    participant_id: str
    language: str = "en"
    room_name: Optional[str]
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
    type: Literal["transcription"] = "transcription"
    participant_id: str
    transcript: str
    is_final: bool
    confidence: float = Field(ge=0.0, le=1.0)
    transcript_id: uuid.UUID


class MeetweenErrorResponse(BaseModel):
    type: Literal["error"] = "error"
    message: str
    code: Optional[str] = None



class TranscriptionResult(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.95)
    language: str = "en"
    error: Optional[str] = None

    @property
    def has_error(self) -> bool:
        return self.error is not None
