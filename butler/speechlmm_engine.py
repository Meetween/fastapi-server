#!/usr/bin/env python3
"""
SpeechLMM Engine - Core model interface for audio transcription and summarization
"""

import torch
from pathlib import Path
from typing import Optional, Union, List, Dict

from speechlmm.model.builder import load_pretrained_model
from speechlmm.conversation import conv_templates
from speechlmm.constants import (
    DEFAULT_AUDIO_INPUT_START_TOKEN,
    DEFAULT_AUDIO_INPUT_END_TOKEN
)


class SpeechLMMEngine:
    """Core SpeechLMM model interface for audio processing"""
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        device: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize SpeechLMM engine.
        
        Args:
            model_path: Path to pretrained model
            device: Device to use ('cuda', 'cpu', or None for auto)
            verbose: Print loading messages
        """
        self.verbose = verbose
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if self.verbose:
            print(f"[SpeechLMM] Loading model from {model_path}")
        
        self.model = load_pretrained_model(model_path, device=self.device)
        self.model.eval()
        
        # Handle pad token (as in inference.py)
        self.model.text_decoder.model.generation_config.pad_token_id = (
            self.model.text_decoder.tokenizer.pad_token_id
        )
        # Handle tokenizer pad side
        self.model.text_decoder.tokenizer.padding_side = "left"
        self.model.config.tokenizer_padding_side = "left"
        
        self.conv_version = self.model.text_decoder.conversation_version
        
        if self.verbose:
            print(f"[SpeechLMM] Model loaded successfully")
    
    def _generate(
        self,
        prompt: str,
        audio: torch.Tensor,
        sample_rate: int,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        repetition_penalty: float = 1.0,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate text from audio.
        
        Args:
            prompt: Task instruction (e.g., "Transcribe the following:")
            audio: Audio tensor
            sample_rate: Audio sample rate
            max_new_tokens: Maximum tokens to generate
            do_sample: Use sampling (False for greedy decoding)
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty
            context: Optional context string to provide additional information
            history: Optional conversation history. Each entry should have 'role' (user/assistant) and 'content' fields.
            
        Returns:
            Generated text
        """
        conv = conv_templates[self.conv_version].copy()
        
        if history:
            for entry in history:
                role = entry.get('role', 'user')
                content = entry.get('content', '')
                if content:
                    if role.lower() in ['user', 'human']:
                        conv.append_message(conv.roles[0], content)
                    elif role.lower() in ['assistant', 'ai', 'bot']:
                        conv.append_message(conv.roles[1], content)
        
        prompt_parts = []
        if context:
            prompt_parts.append(f"Context: {context}")
        prompt_parts.append(prompt)
        full_prompt_text = "\n".join(prompt_parts)
        
        full_prompt = (
            DEFAULT_AUDIO_INPUT_START_TOKEN + 
            DEFAULT_AUDIO_INPUT_END_TOKEN + 
            f"\n{full_prompt_text}"
        )
        
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        
        input_ids = self.model.text_decoder.tokenizer(
            conv.get_prompt(), 
            return_tensors="pt"
        ).input_ids.to(self.model.device)
        
        # Ensure audio is on the same device as the model
        if audio.device != self.model.device:
            audio = audio.to(self.model.device)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                audios=[(audio, sample_rate)],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                repetition_penalty=repetition_penalty if repetition_penalty != 1.0 else None
            )
        
        output_text = self.model.text_decoder.tokenizer.decode(
            output_ids[0], 
            skip_special_tokens=True
        )
        
        return output_text

