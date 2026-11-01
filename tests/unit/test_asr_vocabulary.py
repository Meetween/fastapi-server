"""Tests for the ASR_VOCABULARY setting reaching the transcription prompt."""

import asyncio
from unittest.mock import patch

import numpy as np

from butler.jitsi_api_functions import _transcription_prompt, caption_audio
from config import Settings

BASE_PROMPT = "Can you transcribe this audio?"


class _RecordingEngine:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def _generate_text(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return "hello"


def _pcm_chunk() -> bytes:
    return (np.zeros(4800, dtype=np.int16)).tobytes()


def test_asr_vocabulary_defaults_to_empty() -> None:
    assert Settings(_env_file=None).asr_vocabulary == ""


def test_transcription_prompt_unchanged_without_vocabulary() -> None:
    with patch("butler.jitsi_api_functions.settings.asr_vocabulary", ""):
        assert _transcription_prompt("en-US") == BASE_PROMPT


def test_transcription_prompt_mentions_configured_names() -> None:
    with patch("butler.jitsi_api_functions.settings.asr_vocabulary", "Agentar,Meetween"):
        prompt = _transcription_prompt("en-US")
    assert prompt.startswith(BASE_PROMPT)
    assert "Agentar, Meetween" in prompt


def test_caption_audio_sends_vocabulary_to_the_model() -> None:
    engine = _RecordingEngine()
    with patch("butler.jitsi_api_functions.settings.asr_vocabulary", "Agentar"):
        asyncio.run(caption_audio(engine, _pcm_chunk(), sample_rate=48000))
    with patch("butler.jitsi_api_functions.settings.asr_vocabulary", ""):
        asyncio.run(caption_audio(engine, _pcm_chunk(), sample_rate=48000))
    assert "Agentar" in engine.prompts[0]
    assert engine.prompts[1] == BASE_PROMPT
