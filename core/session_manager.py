import logging
from typing import Dict, Optional
from fastapi import WebSocket
from butler import CaptionHandler, SileroVAD
from butler.triton_client import TritonClient

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
    def add_connection(cls, meeting_id: str, websocket: WebSocket):
        """Add a WebSocket connection"""
        cls._connections[meeting_id] = websocket
        logger.info(f"Added WebSocket connection: {meeting_id}")

    @classmethod
    def remove_connection(cls, meeting_id: str):
        """Remove a WebSocket connection"""
        if meeting_id in cls._connections:
            del cls._connections[meeting_id]
            logger.info(f"Removed WebSocket connection: {meeting_id}")

    @classmethod
    def get_connection(cls, meeting_id: str) -> Optional[WebSocket]:
        """Get a WebSocket connection"""
        return cls._connections.get(meeting_id)

    @classmethod
    async def cleanup(cls):
        """Close all active WebSocket connections"""
        for meeting_id, ws in list(cls._connections.items()):
            try:
                await ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket {meeting_id}: {e}")

        cls._connections.clear()
        cls._caption_handlers.clear()
        logger.info("All connections closed")

    @classmethod
    def get_or_create_caption_handler(
        cls,
        meeting_id: str,
        engine: TritonClient,
        vad_model: Optional[SileroVAD] = None
    ) -> CaptionHandler:
        """Get existing or create new caption handler"""
        if meeting_id not in cls._caption_handlers:
            cls._caption_handlers[meeting_id] = CaptionHandler()
                # engine=engine,
                # vad_model=vad_model
            # )
            logger.info(f"Created caption handler for session: {meeting_id}")

        return cls._caption_handlers[meeting_id]

    @classmethod
    def reset_caption_handler(cls, meeting_id: str) -> bool:
        """Reset a caption handler session"""
        if meeting_id in cls._caption_handlers:
            del cls._caption_handlers[meeting_id]
            logger.info(f"Reset caption handler: {meeting_id}")
            return True
        return False
