#!/usr/bin/env python3
"""
Jitsi API Functions for SpeechLMM Integration

Provides three main functions:
- transcribe_audio(): Full meeting transcription
- caption_audio(): Real-time captioning
- summarize_audio(): Meeting summarization

Audio format: PCM LINEAR16, 48kHz, mono/stereo
"""

import json
from models import TASK_TYPE_TEXT_SUMMARY, TASK_TYPE_SPEECH_SUMMARY, HistoryEntry
import numpy as np
import torch
try:
    import torchaudio  # optional; may be missing or binary-incompatible
except Exception:  # pragma: no cover
    torchaudio = None
import librosa
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from .triton_client import TritonClient
from utils.shared_prompts import PromptBundle


def monify_and_resample_audio(
    audio: torch.Tensor,
    orig_sr: int,
    target_sr: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Convert audio to mono and resample to target sample rate.
    
    Args:
        audio: Audio tensor (shape: [channels, samples] or [1, channels, samples])
        orig_sr: Original sample rate
        target_sr: Target sample rate (default: same as orig_sr)
        
    Returns:
        Tuple of (mono_audio_tensor, target_sr)
    """
    target_sr = target_sr or orig_sr
    # Prefer torchaudio when available; otherwise fallback to librosa.
    if torchaudio is not None:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        audio = resampler(audio)
        return audio.mean(dim=0, keepdim=True), target_sr

    # Fallback: CPU resample via librosa.
    # audio: [channels, samples] or [1, channels, samples]
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    x = audio.detach().cpu().numpy()
    if x.ndim == 1:
        x = x[None, :]
    # Mono first for speed/stability.
    x_mono = x.mean(axis=0)
    y = librosa.resample(x_mono.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
    return y_t, target_sr


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
            prompts_path = Path(__file__).parent / "prompts" / "INPUTS_TEXT_LIST.json"
            with open(prompts_path, 'r', encoding='utf-8') as f:
                cls._prompts = json.load(f)
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


class ResultType(str, Enum):
    """
    Type of transcription result.
    
    - NO_RESULT:     No speech detected
    - PARTIAL_RESULT: Interim / non-final hypothesis for streaming captions
    - FINAL_RESULT:   Final result for a caption segment or full transcription
    """
    NO_RESULT = "NO_RESULT"
    PARTIAL_RESULT = "PARTIAL_RESULT"
    FINAL_RESULT = "FINAL_RESULT"


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
    result_type: ResultType = ResultType.PARTIAL_RESULT


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
        # self.last_was_final = False

    def process_chunk(self, transcription: str) -> Tuple[str, bool, int]:
        # Old logic: determine final by sentence-ending punctuation
        # is_final = self._is_final(transcription)
        # if self.last_was_final:
        #     self.message_id += 1
        # self.last_was_final = is_final
        # return transcription.strip(), is_final, self.message_id

        self.message_id += 1
        return transcription.strip(), True, self.message_id

    # def _is_final(self, text: str) -> bool:
    #     text_stripped = text.strip()
    #     if len(text_stripped) < 5:
    #         return False
    #     sentence_endings = ['.', '!', '?', '。', '！', '？']
    #     return text_stripped[-1] in sentence_endings if text_stripped else False

    def reset(self):
        self.message_id = 0
        # self.last_was_final = False


async def transcribe_audio(
    engine: TritonClient,
    audio_bytes: bytes,
    sample_rate: int = 48000,
    participant_id: Optional[str] = None,
    language: str = "en-US",
    max_new_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
) -> TranscriptionResult:
    """
    Full meeting transcription.
    
    Args:
        engine: Initialized TritonClient
        audio_bytes: Raw PCM LINEAR16 audio bytes (full meeting)
        sample_rate: Input sample rate (Hz)
        participant_id: User ID
        language: BCP-47 language code
        max_tokens: Max generation tokens (model supports up to 131072)
        do_sample: Use sampling (optional)
        temperature: Sampling temperature (optional)
        repetition_penalty: Penalty for repeated tokens (optional)

    Returns:
        TranscriptionResult with complete transcript
    """
    audio_tensor, target_sr = AudioConverter.bytes_to_tensor(audio_bytes, sample_rate)

    prompt = PromptBundle(language=language).transcribe(verbatim=True)

    gen_kwargs: dict = {}
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens
    if do_sample is not None:
        gen_kwargs["do_sample"] = do_sample
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    transcription = await engine._generate_text(
        prompt=prompt,
        audio=audio_tensor,
        sample_rate=target_sr,
        task="transcribe",
        **gen_kwargs,
    )
    
    return TranscriptionResult(
        participant_id=participant_id,
        message_id=0,
        timestamp=0.0,
        is_interim=False,
        language=language,
        stability=1.0,
        text=transcription.strip(),
        confidence=1.0,
        result_type=ResultType.FINAL_RESULT,
    )


async def caption_audio(
    engine: TritonClient,
    audio_bytes: bytes,
    sample_rate: int = 48000,
    participant_id: Optional[str] = None,
    language: str = "en-US",
    caption_handler: Optional[CaptionHandler] = None,
    timestamp: float = 0.0,
    max_tokens: int = 256,
) -> TranscriptionResult:
    """
    Real-time captioning for live transcription display.
    
    Each call transcribes a single chunk independently (no overlap handling).
    The UI should display chunks in a sliding window (like subtitles).
       
    Args:
        engine: Initialized TritonClient
        audio_bytes: Raw PCM LINEAR16 audio bytes (2-second chunk)
        sample_rate: Input sample rate (default: 48000 Hz)
        participant_id: User ID for speaker identification
        language: BCP-47 language code
        caption_handler: CaptionHandler instance (reuse for session)
        timestamp: Audio start time in seconds
        max_tokens: Max generation tokens
        
    Returns:
        TranscriptionResult with current chunk transcription.
        - is_interim: Always False (every chunk is treated as final)
        - stability: Always 1.0
        - message_id: Increments on every chunk
    """
    if caption_handler is None:
        caption_handler = CaptionHandler()
    
    audio_tensor, target_sr = AudioConverter.bytes_to_tensor(audio_bytes, sample_rate)

    prompt = PromptBundle(language=language, caption_chars=60).caption()
    
    transcription = await engine._generate_text(
        prompt=prompt,
        audio=audio_tensor,
        sample_rate=target_sr,
        task="caption",
        max_new_tokens=max_tokens,
        do_sample=False,
        repetition_penalty=1.2,
    )
    
    text, _, message_id = caption_handler.process_chunk(transcription)

    if not text:
        return TranscriptionResult(
            participant_id=participant_id,
            message_id=message_id,
            timestamp=timestamp,
            is_interim=False,
            language=language,
            stability=0.0,
            text="",
            confidence=0.0,
            result_type=ResultType.NO_RESULT,
        )

    return TranscriptionResult(
        participant_id=participant_id,
        message_id=message_id,
        timestamp=timestamp,
        is_interim=False,
        language=language,
        stability=1.0,
        text=text,
        confidence=1.0,
        result_type=ResultType.FINAL_RESULT,
    )


async def summarize_audio(
    engine: TritonClient,
    audio_bytes: bytes,
    sample_rate: int = 48000,
    participant_id: Optional[str] = None,
    language: str = "en-US",
    max_tokens: int = 2048,
    prompt: Optional[str] = None,
    context: Optional[str] = None,
    history: Optional[List[HistoryEntry]] = None
) -> str:
    """
    Meeting summarization.
    
    Args:
        engine: Initialized TritonClient
        audio_bytes: Raw PCM LINEAR16 audio bytes (full meeting/segment)
        sample_rate: Input sample rate (Hz)
        participant_id: User ID
        language: BCP-47 language code
        max_tokens: Max generation tokens (model supports up to 131072)
        prompt: Optional free-form prompt to pass to the LLM. If provided, overrides the default SSUM prompt.
        context: Optional context string to provide additional information to the model
        history: Optional conversation history.
        
    Returns:
        Summary text
    """
    audio_tensor, target_sr = AudioConverter.bytes_to_tensor(audio_bytes, sample_rate)
    
    # Get language-specific SSUM prompt if custom prompt not provided
    if prompt is None:
        prompt = PromptLoader.get_prompt(TASK_TYPE_SPEECH_SUMMARY, language)
    
    summary = await engine._generate_text(
        prompt=prompt,
        audio=audio_tensor,
        sample_rate=target_sr,
        task=TASK_TYPE_SPEECH_SUMMARY,
        max_new_tokens=max_tokens,
        do_sample=False
    )
    return summary.strip()


async def summarize_text(
    engine: TritonClient,
    text: str,
    language: str = "en-US",
    max_new_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
) -> str:
    """
    Summarize text input.
    
    Args:
        engine: Initialized TritonClient
        text: Text to summarize
        language: BCP-47 language code
        max_new_tokens: Max generation tokens
        do_sample: Use sampling
        temperature: Sampling temperature
        repetition_penalty: Penalty for repeated tokens
        
    Returns:
        Summary text
    """
    prompt = PromptLoader.get_prompt(TASK_TYPE_TEXT_SUMMARY, language)
    prompt_with_text = f"{prompt}\n\n{text}"
    summary = await engine._generate_text(
        prompt=prompt_with_text,
        audio=None,
        sample_rate=16000,
        task=TASK_TYPE_TEXT_SUMMARY,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        repetition_penalty=repetition_penalty
    )
    return summary.strip()
