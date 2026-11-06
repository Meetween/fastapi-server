import os
import threading
import logging
from typing import Optional
from config import settings
from butler.triton_client import TritonClient

logger = logging.getLogger(__name__)

class EngineManager:
    """Singleton manager for the primary speechlmm engine and the optional closed-caption engine."""
    _speechlmm_engine: Optional[TritonClient] = None
    _closed_caption_engine: Optional[TritonClient] = None
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
                    prompt_protocol=settings.triton.prompt_protocol,
                    caption_chars=settings.triton.caption_chars,
                )
                logger.info("Triton client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Triton client: {e}", exc_info=True)
                raise RuntimeError(f"Failed to initialize Triton client: {e}")

            cc_settings = settings.closed_caption_triton
            if cc_settings is not None:
                # Optional engine: failure here falls back to the primary engine instead of failing startup.
                try:
                    logger.info(f"Connecting to closed-caption Triton server: {cc_settings.url}")
                    cls._closed_caption_engine = TritonClient(
                        triton_url=cc_settings.url,
                        model_name=cc_settings.model_name,
                        timeout=cc_settings.timeout,
                        verbose=True,
                        max_context_chars=cc_settings.max_context_chars,
                        max_history_chars=cc_settings.max_history_chars,
                        prompt_protocol=cc_settings.prompt_protocol,
                        caption_chars=cc_settings.caption_chars,
                    )
                    logger.info("Closed-caption Triton client initialized successfully")
                except Exception as e:
                    logger.error(
                        f"Failed to initialize closed-caption Triton client, falling back to primary engine: {e}",
                        exc_info=True,
                    )
                    cls._closed_caption_engine = None

            cls._initialized = True
            logger.info("EngineManager initialized")

    @classmethod
    async def cleanup(cls):
        """Cleanup"""
        with cls._lock:
            logger.info("Cleaning up EngineManager...")
            if cls._speechlmm_engine:
                await cls._speechlmm_engine.close()
            if cls._closed_caption_engine:
                await cls._closed_caption_engine.close()
            cls._initialized = False
            cls._speechlmm_engine = None
            cls._closed_caption_engine = None
            logger.info("EngineManager cleanup complete")

    @classmethod
    def get_lmm(cls) -> Optional[TritonClient]:
        """Get the primary SpeechLMM instance"""
        return cls._speechlmm_engine

    @classmethod
    def get_closed_caption_engine(cls) -> Optional[TritonClient]:
        """Engine for the websocket caption path: the closed-caption engine when loaded, else the primary engine."""
        if cls._closed_caption_engine is not None:
            return cls._closed_caption_engine
        return cls.get_lmm()

    @classmethod
    def is_closed_caption_configured(cls) -> bool:
        return settings.closed_caption_triton is not None

    @classmethod
    def is_closed_caption_loaded(cls) -> bool:
        """True when the dedicated closed-caption engine is configured and connected."""
        return cls._closed_caption_engine is not None

    @classmethod
    def is_ready(cls) -> bool:
        """Check if models are ready"""
        return cls._initialized and cls._speechlmm_engine is not None
