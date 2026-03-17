import os
import threading
import logging
from typing import Optional
from config import settings
from butler.triton_client import TritonClient

logger = logging.getLogger(__name__)

class EngineManager:
    """Singleton manager for managing speechlmm."""
    _speechlmm_engine: Optional[TritonClient] = None
    _initialized: bool = False
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        raise RuntimeError("EngineManager can't be instantiated. Use initialize() only.")

    @classmethod
    def initialize(cls):
        with cls._lock:
            if cls._initialized:
                logger.warning("EngineManager already initialized")
                return

            logger.info("Initializing EngineManager...")

            # Initialize Triton client
            try:
                logger.info(f"Connecting to Triton server: {settings.triton.url}")
                cls._speechlmm_engine = TritonClient(
                    triton_url=settings.triton.url,
                    model_name=settings.triton.model_name,
                    timeout=settings.triton.timeout,
                    verbose=True,
                    max_context_chars=settings.triton.max_context_chars,
                    max_history_chars=settings.triton.max_history_chars,
                )
                logger.info("Triton client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Triton client: {e}", exc_info=True)
                raise RuntimeError(f"Failed to initialize Triton client: {e}")

            cls._initialized = True
            logger.info("EngineManager initialized")

    @classmethod
    async def cleanup(cls):
        """Cleanup"""
        with cls._lock:
            logger.info("Cleaning up EngineManager...")
            if cls._speechlmm_engine:
                await cls._speechlmm_engine.close()
            cls._initialized = False
            cls._speechlmm_engine = None
            logger.info("EngineManager cleanup complete")

    @classmethod
    def get_lmm(cls) -> Optional[TritonClient]:
        """Get SpeechLMM instance"""
        return cls._speechlmm_engine

    @classmethod
    def is_ready(cls) -> bool:
        """Check if models are ready"""
        return cls._initialized and cls._speechlmm_engine is not None
