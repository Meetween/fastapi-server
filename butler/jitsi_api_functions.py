#!/usr/bin/env python3
"""
Jitsi API Functions for SpeechLMM Integration

Provides three main functions:
- transcribe_audio(): Full meeting transcription
- caption_audio(): Real-time captioning
- summarize_audio(): Meeting summarization

Audio format: PCM LINEAR16, 48kHz, mono/stereo
"""

from models import TASK_TYPE_TEXT_SUMMARY, TASK_TYPE_SPEECH_SUMMARY, HistoryEntry, MeetweenTranslationResponse
import numpy as np
import torch
try:
    import torchaudio  # optional; may be missing or binary-incompatible
except Exception:  # pragma: no cover
    torchaudio = None
import librosa
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from .triton_client import TritonClient
from utils.prompt_template import get_task_prompt


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

    prompt = get_task_prompt("ASR", language)

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

    prompt = get_task_prompt("ASR", language)

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
        context: Optional context string to provide additional information to the model
        history: Optional conversation history.
        
    Returns:
        Summary text
    """
    audio_tensor, target_sr = AudioConverter.bytes_to_tensor(audio_bytes, sample_rate)
    # Single source of truth: always use language-specific SSUM prompt catalog.
    prompt = get_task_prompt(TASK_TYPE_SPEECH_SUMMARY, language)
    
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

async def translate_audio(
    engine: TritonClient,
    audio_bytes: bytes,
    language: str,
    target_lang: str,
    sample_rate: int = 16000,
    participant_id: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
) -> str:
    """
    Speech to Text translation.

    Args:
        engine: Initialized TritonClient
        audio_bytes: Raw PCM LINEAR16 audio bytes (full meeting/segment)
        language: Language of input text, BCP-47 language code
        target_lang: Translation language, BCP-47 language code
        sample_rate: Input sample rate (Hz)
        participant_id: User ID
        max_tokens: Max generation tokens (model supports up to 131072)
        do_sample: Use sampling (optional)
        temperature: Sampling temperature (optional)
        repetition_penalty: Penalty for repeated tokens (optional)

    Returns:
        Translation of input audio
    """ 

       # Get language-specific MT prompt
    prompt = PromptLoader.get_prompt("MT", language, target_language=target_lang)
    gen_kwargs: dict = {}
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens
    if do_sample is not None:
        gen_kwargs["do_sample"] = do_sample
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    
    audio_tensor, target_sr = AudioConverter.bytes_to_tensor(audio_bytes, sample_rate)

    translation = await engine._generate_text(
        prompt=prompt,
        audio=audio_tensor,
        sample_rate=target_sr,
        task="MT",
        **gen_kwargs,
    )

    return translation.strip()

async def translate_text(
    engine: TritonClient,
    input_text: str,
    language: str,
    target_lang: str,
    participant_id: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
) -> str:
    """
    Text translation.
    
    Args:
        engine: Initialized TritonClient
        input_text: Text to be translated
        language: Language of input text, BCP-47 language code
        target_lang: Translation language, BCP-47 language code
        max_tokens: Max generation tokens (model supports up to 131072)
        do_sample: Use sampling (optional)
        temperature: Sampling temperature (optional)
        repetition_penalty: Penalty for repeated tokens (optional)

    Returns:
        Translation of input text
    """
    # Get language-specific MT prompt
    prompt = PromptLoader.get_prompt("MT", language, target_language=target_lang)
    prompt += f"\n{input_text}"
    gen_kwargs: dict = {}
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens
    if do_sample is not None:
        gen_kwargs["do_sample"] = do_sample
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    translation = await engine._generate_text(
        prompt=prompt,
        audio=None,
        sample_rate=16000,
        task="MT",
        **gen_kwargs,
    )

    return translation.strip()

 