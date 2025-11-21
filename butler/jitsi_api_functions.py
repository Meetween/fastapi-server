#!/usr/bin/env python3
"""
Jitsi API Functions for SpeechLMM Integration

Provides three main functions:
- transcribe_audio(): Full meeting transcription
- caption_audio(): Real-time captioning with optional VAD
- summarize_audio(): Meeting summarization

Audio format: PCM LINEAR16, 48kHz, mono/stereo
"""

import json
import numpy as np
import torch
import torchaudio
from typing import Optional, Tuple
from dataclasses import dataclass
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from .speechlmm_engine import SpeechLMMEngine
from speechlmm.mm_utils import monify_and_resample_audio
from .vad_utils import SileroVAD


class PromptLoader:
    """Loads language-specific prompts from INPUTS_TEXT_LIST.json."""
    
    _prompts = None
    
    # Language code to readable name mapping
    LANGUAGE_NAMES = {
        'en': 'English',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'es': 'Spanish',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ar': 'Arabic',
        'hi': 'Hindi',
    }
    
    @classmethod
    def _load_prompts(cls):
        """Load prompts from JSON file."""
        if cls._prompts is None:
            # Load JSON from speechlmm package resources
            prompts_file = files('speechlmm.dataset').joinpath('INPUTS_TEXT_LIST.json')
            cls._prompts = json.loads(prompts_file.read_text(encoding='utf-8'))
        return cls._prompts
    
    @classmethod
    def _get_language_name(cls, lang_code: str) -> str:
        """Convert language code to readable name."""
        return cls.LANGUAGE_NAMES.get(lang_code.lower(), lang_code.capitalize())
    
    @classmethod
    def _filter_prompts(cls, prompts: list) -> list:
        """
        Filter prompts to exclude those requiring <source-language>.
        Only keep prompts that work with just <target-language> or no placeholders.
        """
        filtered = []
        for prompt in prompts:
            # Skip prompts that require <source-language> (we don't have source language)
            if '<source-language>' in prompt:
                continue
            filtered.append(prompt)
        return filtered
    
    @classmethod
    def get_prompt(cls, task: str, language: str) -> str:
        """
        Get prompt for a specific task and language.
        
        Args:
            task: Task type (ASR, SSUM, etc.)
            language: BCP-47 language code (e.g., "en-US")
            
        Returns:
            First prompt from the filtered list for the task and language,
            with <target-language> substituted with actual language name.
            Falls back to first prompt if filtering removes all options.
        """
        prompts = cls._load_prompts()
        
        # Convert BCP-47 to ISO 639-1 (e.g., "en-US" -> "en")
        lang_code = language.split('-')[0] if '-' in language else language
        
        # Get prompts for task (default to ASR if task not found)
        if task not in prompts:
            # Fallback: use ASR task prompts
            task_prompts = prompts.get('ASR', {})
        else:
            task_prompts = prompts[task]
        
        # Get language-specific prompt (default to English)
        if lang_code in task_prompts:
            lang_prompts = task_prompts[lang_code]
        elif 'en' in task_prompts:
            lang_prompts = task_prompts['en']
        else:
            # Fallback: use first available language
            if task_prompts:
                lang_prompts = list(task_prompts.values())[0]
            else:
                return "Process the following:"
        
        # Handle structured prompts (like SLU)
        if not isinstance(lang_prompts, list):
            return lang_prompts
        
        # Filter out prompts requiring <source-language>
        filtered_prompts = cls._filter_prompts(lang_prompts)
        
        # If filtering removed all prompts, use the first original prompt as fallback
        if not filtered_prompts:
            filtered_prompts = lang_prompts[:1] if lang_prompts else ["Process the following:"]
        
        # Get first prompt from filtered list
        prompt = filtered_prompts[0]
        
        # Substitute <target-language> with actual language name
        target_lang_name = cls._get_language_name(lang_code)
        prompt = prompt.replace('<target-language>', target_lang_name)
        
        return prompt


@dataclass
class TranscriptionResult:
    """Transcription result matching Jigasi format."""
    participant_id: Optional[str]
    message_id: int
    timestamp: float
    is_interim: bool
    language: str
    stability: float
    text: str
    confidence: float


class AudioConverter:
    """Converts PCM LINEAR16 bytes to torch tensors."""
    
    @staticmethod
    def bytes_to_tensor(
        audio_bytes: bytes,
        sample_rate: int = 48000,
        channels: int = 1,
        dtype: str = "int16"
    ) -> Tuple[torch.Tensor, int]:
        """
        Convert PCM bytes to tensor.
        
        Args:
            audio_bytes: Raw PCM LINEAR16 bytes
            sample_rate: Input sample rate (Hz)
            channels: Number of audio channels
            dtype: Audio data type
            
        Returns:
            (audio_tensor, target_sample_rate)
        """
        if dtype == "int16":
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        if channels > 1:
            audio_array = audio_array.reshape(-1, channels).mean(axis=1)
        
        audio_array = audio_array.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
        
        target_sr = 16000
        if sample_rate != target_sr:
            audio_tensor, _ = monify_and_resample_audio(
                audio_tensor, sample_rate, target_sr
            )
        
        return audio_tensor, target_sr


class CaptionHandler:
    """
    Manages caption state for real-time transcription.
    Returns current chunk transcription with message_id tracking.
    """
    
    def __init__(self):
        self.message_id = 0
        self.last_was_final = False
    
    def process_chunk(self, transcription: str) -> Tuple[str, bool, int]:
        """
        Process transcription and determine if it's final.
        
        Args:
            transcription: Current chunk transcription
            
        Returns:
            Tuple of (text, is_final, message_id)
        """
        is_final = self._is_final(transcription)
        
        if self.last_was_final:
            self.message_id += 1
        
        self.last_was_final = is_final
        
        return transcription.strip(), is_final, self.message_id
    
    def _is_final(self, text: str) -> bool:
        """Check if transcription represents a complete sentence."""
        text_stripped = text.strip()
        if len(text_stripped) < 5:
            return False
        
        sentence_endings = ['.', '!', '?', '。', '！', '？']
        return text_stripped[-1] in sentence_endings if text_stripped else False
    
    def reset(self):
        """Reset state for new session."""
        self.message_id = 0
        self.last_was_final = False


def transcribe_audio(
    engine: SpeechLMMEngine,
    audio_bytes: bytes,
    sample_rate: int = 48000,
    participant_id: Optional[str] = None,
    language: str = "en-US",
    max_tokens: int = 8192
) -> TranscriptionResult:
    """
    Full meeting transcription.
    
    Args:
        engine: Initialized SpeechLMMEngine
        audio_bytes: Raw PCM LINEAR16 audio bytes (full meeting)
        sample_rate: Input sample rate (Hz)
        participant_id: User ID
        language: BCP-47 language code
        max_tokens: Max generation tokens (model supports up to 131072)
        
    Returns:
        TranscriptionResult with complete transcript
    """
    audio_tensor, target_sr = AudioConverter.bytes_to_tensor(audio_bytes, sample_rate)
    
    # Get language-specific ASR prompt
    prompt = PromptLoader.get_prompt("ASR", language)
    
    transcription = engine._generate(
        prompt=prompt,
        audio=audio_tensor,
        sample_rate=target_sr,
        max_new_tokens=max_tokens,
        do_sample=False
    )
    
    return TranscriptionResult(
        participant_id=participant_id,
        message_id=0,
        timestamp=0.0,
        is_interim=False,
        language=language,
        stability=1.0,
        text=transcription.strip(),
        confidence=1.0
    )


def caption_audio(
    engine: SpeechLMMEngine,
    audio_bytes: bytes,
    sample_rate: int = 48000,
    participant_id: Optional[str] = None,
    language: str = "en-US",
    caption_handler: Optional[CaptionHandler] = None,
    timestamp: float = 0.0,
    max_tokens: int = 256,
    vad: Optional[SileroVAD] = None
) -> TranscriptionResult:
    """
    Real-time captioning for live transcription display.
    
    Each call transcribes a single chunk independently (no overlap handling).
    The UI should display chunks in a sliding window (like subtitles).
    
    Args:
        engine: Initialized SpeechLMMEngine
        audio_bytes: Raw PCM LINEAR16 audio bytes (2-second chunk)
        sample_rate: Input sample rate (default: 48000 Hz)
        participant_id: User ID for speaker identification
        language: BCP-47 language code
        caption_handler: CaptionHandler instance (reuse for session)
        timestamp: Audio start time in seconds
        max_tokens: Max generation tokens
        vad: Optional SileroVAD instance for silence detection
        
    Returns:
        TranscriptionResult with current chunk transcription.
        - is_interim: True if chunk doesn't end with punctuation
        - is_final: True if chunk ends with sentence punctuation
        - message_id: Increments each time is_final=True
        - text: Empty string if VAD detects silence
    """
    if caption_handler is None:
        caption_handler = CaptionHandler()
    
    audio_tensor, target_sr = AudioConverter.bytes_to_tensor(audio_bytes, sample_rate)
    
    # VAD: Skip transcription if no speech detected
    if vad is not None and not vad.has_speech(audio_tensor):
        return TranscriptionResult(
            participant_id=participant_id,
            message_id=caption_handler.message_id,
            timestamp=timestamp,
            is_interim=True,
            language=language,
            stability=0.0,
            text="",
            confidence=0.0
        )
    
    # Get language-specific ASR prompt
    prompt = PromptLoader.get_prompt("ASR", language)
    
    transcription = engine._generate(
        prompt=prompt,
        audio=audio_tensor,
        sample_rate=target_sr,
        max_new_tokens=max_tokens,
        do_sample=False,
        repetition_penalty=1.2
    )
    
    text, is_final, message_id = caption_handler.process_chunk(transcription)
    stability = 0.8 if is_final else 0.6
    
    return TranscriptionResult(
        participant_id=participant_id,
        message_id=message_id,
        timestamp=timestamp,
        is_interim=not is_final,
        language=language,
        stability=stability,
        text=text,
        confidence=stability
    )


def summarize_audio(
    engine: SpeechLMMEngine,
    audio_bytes: bytes,
    sample_rate: int = 48000,
    participant_id: Optional[str] = None,
    language: str = "en-US",
    max_tokens: int = 2048
) -> str:
    """
    Meeting summarization.
    
    Args:
        engine: Initialized SpeechLMMEngine
        audio_bytes: Raw PCM LINEAR16 audio bytes (full meeting/segment)
        sample_rate: Input sample rate (Hz)
        participant_id: User ID
        language: BCP-47 language code
        max_tokens: Max generation tokens (model supports up to 131072)
        
    Returns:
        Summary text
    """
    audio_tensor, target_sr = AudioConverter.bytes_to_tensor(audio_bytes, sample_rate)
    
    # Get language-specific SSUM prompt
    prompt = PromptLoader.get_prompt("SSUM", language)
    
    summary = engine._generate(
        prompt=prompt,
        audio=audio_tensor,
        sample_rate=target_sr,
        max_new_tokens=max_tokens,
        do_sample=False
    )
    
    return summary.strip()



