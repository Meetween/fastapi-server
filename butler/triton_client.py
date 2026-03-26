#!/usr/bin/env python3
"""
Triton Client - Interface to Triton Inference Server for SpeechLMM
"""

import base64
import io
import logging
from typing import Optional

import httpx
import torch
import soundfile as sf

logger = logging.getLogger(__name__)

class TritonClient:
    """Client for Triton Inference Server with SpeechLMM model interface"""
    
    def __init__(
        self,
        triton_url: str = "http://localhost:9000",
        model_name: str = "speechlmm",
        timeout: int = 1800,
        verbose: bool = False,
        max_context_chars: Optional[int] = 300_000,
        max_history_chars: Optional[int] = 600_000,
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
        
        self.inference_url = f"{self.triton_url}/v2/models/{self.model_name}/infer"
        
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )
        
        if self.verbose:
            logger.info(f"[TritonClient] Initialized with URL: {self.triton_url}")
            logger.info(f"[TritonClient] Model: {self.model_name}")
    
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
    
    async def _generate_text_from_audio(
        self,
        prompt: str,
        audio: torch.Tensor,
        sample_rate: int,
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
            repetition_penalty: Penalty for repeated tokens (optional)

        Returns:
            Generated text
        """
        wav_bytes = self._tensor_to_wav_bytes(audio, sample_rate)
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')

        # Always send all inputs; use defaults when caller doesn't specify (Triton expects all)
        # For batched model, scalar inputs use shape [1, 1] (batch_dim=1, dims[1]=1)
        inputs = [
            {"name": "AUDIO_BYTES", "shape": [1, 1], "datatype": "BYTES", "data": [audio_b64]},
            {"name": "TEXT_PROMPT", "shape": [1, 1], "datatype": "BYTES", "data": [prompt]},
            {"name": "MAX_NEW_TOKENS", "shape": [1, 1], "datatype": "INT32", "data": [max_new_tokens or 8192]},
            {"name": "DO_SAMPLE", "shape": [1, 1], "datatype": "BOOL", "data": [do_sample if do_sample is not None else False]},
            {"name": "REPETITION_PENALTY", "shape": [1, 1], "datatype": "FP32", "data": [repetition_penalty if repetition_penalty is not None else 1.0]},
            {"name": "TEMPERATURE", "shape": [1, 1], "datatype": "FP32", "data": [temperature if (temperature is not None and (do_sample or False)) else 1.0]},
        ]

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

    async def generate_text_from_text(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """
        Generate text from a raw prompt. Prompt composition is handled by endpoints.
        """
        sample_rate = 16000
        duration_sec = 0.1
        num_samples = int(sample_rate * duration_sec)
        silent_audio = torch.zeros(1, num_samples)
        return await self._generate_text_from_audio(
            prompt=prompt,
            audio=silent_audio,
            sample_rate=sample_rate,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

