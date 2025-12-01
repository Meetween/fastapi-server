#!/usr/bin/env python3
"""
Voice Activity Detection utilities for SpeechLMM
"""

import torch


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
            with torch.no_grad():
                confidence = self.model(audio, self.sample_rate).item()
            return confidence > self.threshold
        except Exception:
            return True
    
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
            with torch.no_grad():
                confidence = self.model(audio, self.sample_rate).item()
            return confidence
        except Exception:
            return 1.0

