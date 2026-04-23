"""
Butler - SpeechLMM integration for Jitsi Meet

Provides audio transcription, captioning, and summarization capabilities.
"""

from .jitsi_api_functions import (
    transcribe_audio,
    caption_audio,
    summarize_audio,
    summarize_text,
    TranscriptionResult,
    CaptionHandler,
    AudioConverter,
    ResultType,
)

__all__ = [
    'transcribe_audio',
    'caption_audio',
    'summarize_audio',
    'summarize_text',
    'TranscriptionResult',
    'CaptionHandler',
    'AudioConverter',
    'ResultType',
]

