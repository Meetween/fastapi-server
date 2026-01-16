#!/usr/bin/env python3
"""
Triton Client - Interface to Triton Inference Server for SpeechLMM
"""

import base64
import io
import logging
import torch
import soundfile as sf
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TritonClient:
    """Client for Triton Inference Server with SpeechLMM model interface"""
    
    def __init__(
        self,
        triton_url: str = "http://localhost:9000",
        model_name: str = "speechlmm",
        timeout: int = 1800,
        verbose: bool = False
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
        
        self.inference_url = f"{self.triton_url}/v2/models/{self.model_name}/infer"
        
        if self.verbose:
            logger.info(f"[TritonClient] Initialized with URL: {self.triton_url}")
            logger.info(f"[TritonClient] Model: {self.model_name}")
    
    def _tensor_to_wav_bytes(self, audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
        """Convert audio tensor to WAV bytes."""
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze(0)
        
        audio_numpy = audio_tensor.cpu().numpy()
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_numpy, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()
    
    def _generate(
        self,
        prompt: str,
        audio: torch.Tensor,
        sample_rate: int,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        repetition_penalty: float = 1.0
    ) -> str:
        """
        Generate text from audio using Triton server.
        
        Args:
            prompt: Task instruction
            audio: Audio tensor
            sample_rate: Audio sample rate
            max_new_tokens: Maximum tokens to generate
            do_sample: Use sampling (ignored, Triton uses greedy)
            temperature: Sampling temperature (ignored)
            repetition_penalty: Repetition penalty (ignored)
            
        Returns:
            Generated text
        """
        wav_bytes = self._tensor_to_wav_bytes(audio, sample_rate)
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        payload = {
            "inputs": [
                {
                    "name": "AUDIO_BYTES",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [[audio_b64]]
                },
                {
                    "name": "TEXT_PROMPT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [[prompt]]
                }
            ],
            "outputs": [{"name": "GENERATED_TEXT"}]
        }
        
        if self.verbose:
            logger.info(f"[TritonClient] Sending request: {len(wav_bytes)} bytes, prompt: {prompt[:50]}...")
        
        try:
            response = requests.post(
                self.inference_url,
                json=payload,
                timeout=self.timeout
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
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[TritonClient] Request failed: {e}")
            raise RuntimeError(f"Triton inference failed: {str(e)}")
        except Exception as e:
            logger.error(f"[TritonClient] Error processing response: {e}")
            raise RuntimeError(f"Triton response processing failed: {str(e)}")

