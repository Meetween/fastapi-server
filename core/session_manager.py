import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fastapi import WebSocket

from butler import CaptionHandler
from butler.triton_client import TritonClient
from models import HistoryEntry

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """
    In-memory session state for a meeting.

    - `meeting_id` / `room_name`: identify the active meeting.
    - `transcript`: running transcript used as context for Q&A.
    - `participant_histories`: per-participant chat history (HistoryEntry list).
    """

    meeting_id: str
    room_name: Optional[str] = None
    transcript: Optional[str] = None
    participant_histories: Dict[str, List[HistoryEntry]] = field(default_factory=dict)


class SessionManager:
    """Manages WebSocket connections, caption handlers, and meeting sessions"""

    _instance = None

    _connections: Dict[str, WebSocket] = {}
    _caption_handlers: Dict[str, CaptionHandler] = {}
    _sessions: Dict[str, Session] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance

    # WebSocket connection management

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
        cls._sessions.clear()
        logger.info("All connections closed")

    @classmethod
    def get_or_create_caption_handler(
        cls,
        meeting_id: str,
        engine: TritonClient,
    ) -> CaptionHandler:
        """Get existing or create new caption handler"""
        if meeting_id not in cls._caption_handlers:
            cls._caption_handlers[meeting_id] = CaptionHandler()
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

    # Session and chat history management
    @classmethod
    def get_or_create_session(cls, meeting_id: str, room_name: Optional[str] = None) -> Session:
        """
        Get existing session for a meeting or create a new one.

        The session stores the full transcript and per-participant chat history
        in memory for the lifetime of the meeting.
        """
        session = cls._sessions.get(meeting_id)
        if session is None:
            session = Session(meeting_id=meeting_id, room_name=room_name)
            cls._sessions[meeting_id] = session
            logger.info(f"Created session: {meeting_id}")
        elif room_name and not session.room_name:
            session.room_name = room_name
        return session

    @classmethod
    def update_transcript(cls, meeting_id: str, room_name: Optional[str], transcript: Optional[str]) -> None:
        """Update the stored transcript for a meeting."""
        session = cls.get_or_create_session(meeting_id=meeting_id, room_name=room_name)
        session.transcript = f"{session.transcript or ''} {transcript}".strip()

    @classmethod
    def get_transcript(cls, meeting_id: str, room_name: Optional[str]) -> Optional[str]:
        """Return the current transcript for a meeting, if any."""
        session = cls.get_or_create_session(meeting_id=meeting_id, room_name=room_name)
        return session.transcript

    @classmethod
    def get_participant_history(cls, meeting_id: str, room_name: Optional[str], participant_id: str) -> List[HistoryEntry]:
        """
        Get the in-memory chat history for a participant in a meeting.

        Returns an empty list if no history is stored yet.
        """
        session = cls.get_or_create_session(meeting_id=meeting_id, room_name=room_name)
        return session.participant_histories.get(participant_id, [])

    @classmethod
    def store_chat_turn(
        cls,
        meeting_id: str,
        room_name: Optional[str],
        participant_id: str,
        user_prompt: str,
        assistant_reply: str,
    ) -> None:
        """
        Store a single Q&A turn for a participant.
        """
        session = cls.get_or_create_session(meeting_id=meeting_id, room_name=room_name)
        if participant_id not in session.participant_histories:
            session.participant_histories[participant_id] = []
        session.participant_histories[participant_id].extend([
            HistoryEntry(role="user", content=user_prompt),
            HistoryEntry(role="assistant", content=assistant_reply),
        ])
        logger.debug(
            "Stored chat turn for meeting_id=%s participant_id=%s (participant_histories=%d)",
            meeting_id,
            participant_id,
            len(session.participant_histories[participant_id]),
        )

    @classmethod
    def clear_session(cls, meeting_id: str, room_name: Optional[str]) -> None:
        """Remove all in-memory state for a meeting."""
        session = cls.get_or_create_session(meeting_id=meeting_id, room_name=room_name)
        session.participant_histories.clear()
        session.transcript = None
        logger.info(f"Cleared session: {meeting_id} room_name={room_name}")
        cls._sessions.pop(meeting_id, None)