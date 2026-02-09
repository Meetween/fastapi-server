import os
import threading
import logging
from typing import Optional
from config import settings
from butler import SpeechLMMEngine, SileroVAD

logger = logging.getLogger(__name__)

class EngineManager:
    """
    Singleton manager for managing speechlmm engines and vad_model.
    
    Maintains separate engine instances:
    - Real-time engine: For websocket captioning and transcription
    - Batch engine: For catchup and summarization
    """
    _realtime_engine: Optional[SpeechLMMEngine] = None
    _batch_engine: Optional[SpeechLMMEngine] = None
    _vad_model: Optional[SileroVAD] = None
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

            # Validate model path
            model_path = settings.speechlmm_model_path
            if not model_path:
                raise RuntimeError("SPEECHLMM_MODEL_PATH environment variable not set")

            if not os.path.exists(model_path):
                raise RuntimeError(f"Model path not found: {model_path}")

            try:
                logger.info(f"Loading real-time SpeechLMM model from {model_path}")
                cls._realtime_engine = SpeechLMMEngine(
                    model_path=model_path,
                    device=None,
                    verbose=True
                )
                logger.info("Real-time SpeechLMM model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load real-time SpeechLMM model: {e}", exc_info=True)
                raise RuntimeError(f"Failed to initialize real-time SpeechLMM: {e}")

            try:
                logger.info(f"Loading batch SpeechLMM model from {model_path}")
                cls._batch_engine = SpeechLMMEngine(
                    model_path=model_path,
                    device=None,
                    verbose=True
                )
                logger.info("Batch SpeechLMM model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load batch SpeechLMM model: {e}", exc_info=True)
                raise RuntimeError(f"Failed to initialize batch SpeechLMM: {e}")

            # Load VAD model
            try:
                cls._vad_model = SileroVAD(
                    sample_rate=settings.vad_sample_rate,
                    threshold=settings.vad_threshold
                )
                logger.info("VAD model loaded successfully")
            except Exception as e:
                cls._vad_model = None
                raise RuntimeError(f"Failed to load VAD model: {e}")

            cls._initialized = True
            logger.info("EngineManager initialized")

    @classmethod
    async def cleanup(cls):
        """Cleanup"""
        with cls._lock:
            logger.info("Cleaning up EngineManager...")
            cls._initialized = False
            cls._realtime_engine = None
            cls._batch_engine = None
            cls._vad_model = None
            logger.info("EngineManager cleanup complete")

    @classmethod
    def get_lmm(cls) -> Optional[SpeechLMMEngine]:
        """
        Get SpeechLMM instance.
        Returns real-time engine by default.
        """
        return cls._realtime_engine

    @classmethod
    def get_realtime_lmm(cls) -> Optional[SpeechLMMEngine]:
        """
        Get real-time SpeechLMM engine instance.
        Use for websocket captioning and transcription.
        """
        return cls._realtime_engine

    @classmethod
    def get_batch_lmm(cls) -> Optional[SpeechLMMEngine]:
        """
        Get batch SpeechLMM engine instance.
        Use for catchup and summarization.
        """
        return cls._batch_engine

    @classmethod
    def get_vad(cls) -> Optional[SileroVAD]:
        """Get VAD model instance"""
        return cls._vad_model

    @classmethod
    def is_ready(cls) -> bool:
        """Check if models are ready"""
        return cls._initialized and cls._realtime_engine is not None and cls._batch_engine is not None
