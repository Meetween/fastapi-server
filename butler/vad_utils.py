#!/usr/bin/env python3
"""
Voice Activity Detection utilities for SpeechLMM
"""

import logging
import torch

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Lightweight Voice Activity Detection using Silero VAD.
    Detects speech presence in audio to skip silence during real-time captioning.
    """
    
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        """
        Initialize VAD.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            threshold: Speech detection threshold [0.0-1.0]
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model from torch.hub"""
        try:
            self.model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                verbose=False
            )
            self.model.eval()
        except Exception as e:
            print(f"[VAD] Warning: Failed to load Silero VAD: {e}")
            self.model = None
    
    def has_speech(self, audio: torch.Tensor) -> bool:
        """
        Detect if audio contains speech.
        
        Args:
            audio: Audio tensor [1, samples] or [samples]
            
        Returns:
            True if speech detected, False if silence
        """
        if self.model is None:
            return True
        
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        
        # Too short to analyze
        if audio.shape[0] < self.sample_rate * 0.25:
            return True
        
        try:
            # Silero VAD expects chunks of 256 samples (8kHz) or 512 samples (16kHz)
            chunk_size = 512 if self.sample_rate == 16000 else 256
            
            max_confidence = 0.0
            num_samples = audio.shape[0]
            
            for start_idx in range(0, num_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, num_samples)
                chunk = audio[start_idx:end_idx]
                
                if chunk.shape[0] < chunk_size:
                    padding = chunk_size - chunk.shape[0]
                    chunk = torch.nn.functional.pad(chunk, (0, padding))
                
                with torch.no_grad():
                    chunk_confidence = self.model(chunk, self.sample_rate).item()
                    max_confidence = max(max_confidence, chunk_confidence)
            
            return max_confidence > self.threshold
        except Exception as e:
            logger.error(f"Error detecting speech: {e}")
            return False
    
    def get_confidence(self, audio: torch.Tensor) -> float:
        """
        Get speech confidence score.
        
        Args:
            audio: Audio tensor [1, samples] or [samples]
            
        Returns:
            Confidence score [0.0-1.0], where 1.0 = high confidence speech
        """
        if self.model is None:
            return 1.0
        
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        
        try:
            # Silero VAD expects chunks of 256 samples (8kHz) or 512 samples (16kHz)
            chunk_size = 512 if self.sample_rate == 16000 else 256
            
            # Process audio in chunks
            max_confidence = 0.0
            num_samples = audio.shape[0]
            
            # Process chunks
            for start_idx in range(0, num_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, num_samples)
                chunk = audio[start_idx:end_idx]
                
                # Pad chunk if it's shorter than required size
                if chunk.shape[0] < chunk_size:
                    padding = chunk_size - chunk.shape[0]
                    chunk = torch.nn.functional.pad(chunk, (0, padding))
                
                with torch.no_grad():
                    chunk_confidence = self.model(chunk, self.sample_rate).item()
                    max_confidence = max(max_confidence, chunk_confidence)
            
            return max_confidence
        except Exception as e:
            logger.error(f"Error getting confidence: {e}")
            return 0.0

