"""
Butler - SpeechLMM integration for Jitsi Meet

Provides audio transcription, captioning, and summarization capabilities.
"""

from .jitsi_api_functions import (
    transcribe_audio,
    caption_audio,
    summarize_audio,
    TranscriptionResult,
    CaptionHandler,
    AudioConverter,
    ResultType,
)
from .speechlmm_engine import SpeechLMMEngine
from .vad_utils import SileroVAD

__all__ = [
    'transcribe_audio',
    'caption_audio',
    'summarize_audio',
    'TranscriptionResult',
    'CaptionHandler',
    'AudioConverter',
    'ResultType',
    'SpeechLMMEngine',
    'SileroVAD'
]

