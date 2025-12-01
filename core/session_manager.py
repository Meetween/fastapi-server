import logging
from typing import Dict, Optional
from fastapi import WebSocket
from butler import CaptionHandler, SpeechLMMEngine, SileroVAD

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages WebSocket connections and caption handler sessions"""

    _instance = None


    _connections: Dict[str, WebSocket] = {}
    _caption_handlers: Dict[str, CaptionHandler] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def add_connection(cls, room_name: str, websocket: WebSocket):
        """Add a WebSocket connection"""
        cls._connections[room_name] = websocket
        logger.info(f"Added WebSocket connection: {room_name}")

    @classmethod
    def remove_connection(cls, room_name: str):
        """Remove a WebSocket connection"""
        if room_name in cls._connections:
            del cls._connections[room_name]
            logger.info(f"Removed WebSocket connection: {room_name}")

    @classmethod
    def get_connection(cls, room_name: str) -> Optional[WebSocket]:
        """Get a WebSocket connection"""
        return cls._connections.get(room_name)

    @classmethod
    async def cleanup(cls):
        """Close all active WebSocket connections"""
        for room_name, ws in list(cls._connections.items()):
            try:
                await ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket {room_name}: {e}")

        cls._connections.clear()
        cls._caption_handlers.clear()
        logger.info("All connections closed")

    @classmethod
    def get_or_create_caption_handler(
        cls,
        room_name: str,
        engine: SpeechLMMEngine,
        vad_model: Optional[SileroVAD] = None
    ) -> CaptionHandler:
        """Get existing or create new caption handler"""
        if room_name not in cls._caption_handlers:
            cls._caption_handlers[room_name] = CaptionHandler()
                # engine=engine,
                # vad_model=vad_model
            # )
            logger.info(f"Created caption handler for session: {room_name}")

        return cls._caption_handlers[room_name]

    @classmethod
    def reset_caption_handler(cls, room_name: str) -> bool:
        """Reset a caption handler session"""
        if room_name in cls._caption_handlers:
            del cls._caption_handlers[room_name]
            logger.info(f"Reset caption handler: {room_name}")
            return True
        return False
