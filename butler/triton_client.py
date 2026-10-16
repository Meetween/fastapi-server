#!/usr/bin/env python3
"""
Triton Client - Interface to Triton Inference Server for SpeechLMM
"""

import base64
import io
import logging
from typing import Optional
import json

import httpx
import torch
import soundfile as sf

from models import TaskType

logger = logging.getLogger(__name__)

# Modes that accept a source language in the general_json payload.
_LANGUAGE_MODES = {
    t.lower()
    for t in (
        TaskType.CAPTION,
        TaskType.TRANSCRIPTION,
        TaskType.SPEECH_TRANSLATION,
        TaskType.TEXT_TRANSLATION,
    )
}

class TritonClient:
    """Client for Triton Inference Server with SpeechLMM model interface"""
    
    def __init__(
        self,
        triton_url: str = "http://localhost:9000",
        model_name: str = "qwen25_omni",
        timeout: int = 1800,
        verbose: bool = False,
        max_context_chars: Optional[int] = 300_000,
        max_history_chars: Optional[int] = 600_000,
        prompt_protocol: str = "speechlmm_plain",
        caption_chars: int = 60,
    ):
        """
        Initialize Triton client.
        
        Args:
            triton_url: Triton server URL
            model_name: Model name in Triton
            timeout: Request timeout in seconds
            verbose: Print debug messages
        """
        self.triton_url = triton_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.verbose = verbose
        self.device = "cuda"
        self.max_context_chars = max_context_chars
        self.max_history_chars = max_history_chars
        self.prompt_protocol = (prompt_protocol or "speechlmm_plain").strip().lower()
        self.caption_chars = int(caption_chars) if caption_chars is not None else 60
        
        self.inference_url = f"{self.triton_url}/v2/models/{self.model_name}/infer"
        
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )
        
        if self.verbose:
            logger.info(f"[TritonClient] Initialized with URL: {self.triton_url}")
            logger.info(f"[TritonClient] Model: {self.model_name}")
            logger.info(f"[TritonClient] Prompt protocol: {self.prompt_protocol}")
    
    async def close(self):
        """Close the HTTP client connection pool"""
        await self._client.aclose()
    
    def _tensor_to_wav_bytes(self, audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
        """Convert audio tensor to WAV bytes."""
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze(0)
        
        audio_numpy = audio_tensor.cpu().numpy()
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_numpy, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()
    
    async def _generate_text(
        self,
        prompt: str,
        audio: torch.Tensor,
        sample_rate: int,
        task: Optional[str] = None,
        language: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """
        Generate text from audio using Triton server.

        Optional params (max_new_tokens, do_sample, temperature, repetition_penalty):
        when provided, sent to Triton; when None, omitted and Triton uses its defaults.

        Args:
            prompt: Task instruction
            audio: Audio tensor
            sample_rate: Audio sample rate
            max_new_tokens: Max tokens to generate (optional)
            do_sample: Use sampling (optional)
            temperature: Sampling temperature when do_sample=True (optional)
            repetition_penalty: Accepted for compatibility but ignored

        Returns:
            Generated text
        """
        wav_bytes = b""
        audio_b64 = None
        if audio is not None:
            wav_bytes = self._tensor_to_wav_bytes(audio, sample_rate)
            audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        elif self.prompt_protocol != "general_json":
            # SpeechLMM Triton requires AUDIO_BYTES on every request; use brief silence for text-only.
            silent = torch.zeros(1600, dtype=torch.float32)
            wav_bytes = self._tensor_to_wav_bytes(silent, 16000)
            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        # Optional adapter for models that support a structured prompt protocol.
        # SpeechLMM expects a plain string prompt.
        prompt_to_send = prompt
        if self.prompt_protocol == "general_json":
            mode = (task or "").strip().lower()
            if not mode:
                # Infer best-effort from whether audio is present.
                mode = (
                    TaskType.SPOKEN_QUESTION_ANSWERING if audio_b64 is None else TaskType.TRANSCRIPTION
                ).lower()
            payload_obj: dict = {"mode": mode, "prompt": prompt_to_send, "format": "verbatim"}
            if mode == TaskType.CAPTION.lower():
                payload_obj["caption_chars"] = self.caption_chars
            # Voxtral/Qwen: source language in JSON (caption/transcribe/ST/MT).
            if language and mode in _LANGUAGE_MODES:
                lang_code = language.split("-")[0].strip().lower()
                if lang_code:
                    payload_obj["language"] = lang_code
            prompt_to_send = json.dumps(payload_obj, ensure_ascii=False)

        # Always send all inputs; use defaults when caller doesn't specify (Triton expects all)
        # For batched model, scalar inputs use shape [1, 1] (batch_dim=1, dims[1]=1)
        inputs = [
            {"name": "TEXT_PROMPT", "shape": [1, 1], "datatype": "BYTES", "data": [[prompt_to_send]]},
            {"name": "MAX_NEW_TOKENS", "shape": [1, 1], "datatype": "INT32", "data": [[max_new_tokens or 8192]]},
            {"name": "DO_SAMPLE", "shape": [1, 1], "datatype": "BOOL", "data": [[do_sample if do_sample is not None else False]]},
            {"name": "REPETITION_PENALTY", "shape": [1, 1], "datatype": "FP32", "data": [[repetition_penalty if repetition_penalty is not None else 1.0]]},
            {"name": "TEMPERATURE", "shape": [1, 1], "datatype": "FP32", "data": [[temperature if (temperature is not None and (do_sample or False)) else 1.0]]},
        ]

        if audio_b64 is not None:
            inputs.insert(
                0,
                {"name": "AUDIO_BYTES", "shape": [1, 1], "datatype": "BYTES", "data": [[audio_b64]]},
            )

        payload = {
            "inputs": inputs,
            "outputs": [{"name": "GENERATED_TEXT"}]
        }
        
        if self.verbose:
            logger.info(f"[TritonClient] Sending request: {len(wav_bytes)} bytes, prompt: {prompt[:50]}...")
        
        try:
            response = await self._client.post(
                self.inference_url,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            output = result.get("outputs", [{}])[0]
            data = output.get("data", [])
            
            if not data:
                raise ValueError("No output data in Triton response")
            
            text = data[0] if not isinstance(data[0], list) else data[0][0]
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            
            if self.verbose:
                logger.info(f"[TritonClient] Received response: {len(text)} characters")
            
            return text
            
        except httpx.RequestError as e:
            logger.error(f"[TritonClient] Request failed: {e}")
            raise RuntimeError(f"Triton inference failed: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"[TritonClient] HTTP error: {e}")
            raise RuntimeError(f"Triton inference failed: {str(e)}")
        except Exception as e:
            logger.error(f"[TritonClient] Error processing response: {e}")
            raise RuntimeError(f"Triton response processing failed: {str(e)}")
            