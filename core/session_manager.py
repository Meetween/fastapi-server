import logging
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fastapi import WebSocket

from config import settings
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
    HISTORY_TURNS_LIMIT: int = settings.triton.max_history_turns
    TRANSCRIPT_CHARS_LIMIT: int = settings.triton.max_context_chars

    _instance = None

    _connections: Dict[str, WebSocket] = {}
    _caption_handlers: Dict[str, CaptionHandler] = {}
    _sessions: Dict[str, Session] = {}
    _participant_locks: Dict[str, asyncio.Lock] = {}
    _participant_locks_guard: Optional[asyncio.Lock] = None

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
    def remove_connection(cls, meeting_id: str) -> None:
        """
        Remove a WebSocket connection for a meeting.
        """
        if meeting_id in cls._connections:
            del cls._connections[meeting_id]
            logger.info(f"Removed WebSocket connection: {meeting_id}")

    @classmethod
    def get_connection(cls, meeting_id: str) -> Optional[WebSocket]:
        """Get one WebSocket connection (if any) for a meeting."""
        return cls._connections.get(meeting_id)

    @classmethod
    def clear_meeting_state_if_inactive(
        cls,
        meeting_id: str,
        room_name: Optional[str],
    ) -> bool:
        """
        Clear meeting-level in-memory state when no websocket clients remain.

        Returns True when cleanup was performed, False when the meeting is still active.
        """
        if meeting_id in cls._connections:
            return False

        cls.reset_caption_handler(meeting_id=meeting_id)
        cls.clear_session(meeting_id=meeting_id, room_name=room_name, force=True)
        return True

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
        async with cls._get_participant_locks_guard():
            cls._participant_locks.clear()
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

    @classmethod
    def _participant_lock_key(cls, meeting_id: str, room_name: Optional[str], participant_id: str) -> str:
        return f"{meeting_id}:{room_name or ''}:{participant_id}"

    @classmethod
    def _get_participant_locks_guard(cls) -> asyncio.Lock:
        if cls._participant_locks_guard is None:
            cls._participant_locks_guard = asyncio.Lock()
        return cls._participant_locks_guard

    @classmethod
    async def _get_or_create_participant_lock(
        cls,
        meeting_id: str,
        room_name: Optional[str],
        participant_id: str,
    ) -> asyncio.Lock:
        key = cls._participant_lock_key(meeting_id=meeting_id, room_name=room_name, participant_id=participant_id)
        guard = cls._get_participant_locks_guard()
        async with guard:
            lock = cls._participant_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                cls._participant_locks[key] = lock
            return lock

    @classmethod
    @asynccontextmanager
    async def participant_history_mutex(
        cls,
        meeting_id: str,
        room_name: Optional[str],
        participant_id: str,
    ):
        """
        Serialize operations that depend on a consistent participant history.

        Intended usage is:
        - snapshot participant history (and any other dependent reads)
        - run expensive inference outside the mutex (optional but recommended)
        - store the resulting turn under the mutex
        """
        lock = await cls._get_or_create_participant_lock(
            meeting_id=meeting_id,
            room_name=room_name,
            participant_id=participant_id,
        )
        async with lock:
            yield

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
        """Append transcript text for a meeting"""
        session = cls.get_or_create_session(meeting_id=meeting_id, room_name=room_name)
        if not transcript:
            return

        session.transcript = f"{session.transcript or ''} {transcript}".strip()

        if len(session.transcript) > cls.TRANSCRIPT_CHARS_LIMIT:
            session.transcript = session.transcript[-cls.TRANSCRIPT_CHARS_LIMIT :]

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

        max_entries = cls.HISTORY_TURNS_LIMIT * 2
        entries = session.participant_histories[participant_id]
        if len(entries) > max_entries:
            session.participant_histories[participant_id] = entries[-max_entries:]

        logger.debug(
            "Stored chat turn for meeting_id=%s participant_id=%s (participant_histories=%d)",
            meeting_id,
            participant_id,
            len(session.participant_histories[participant_id]),
        )

    @classmethod
    def clear_session(cls, meeting_id: str, room_name: Optional[str], force: bool = False) -> None:
        """
        Remove all in-memory state for a meeting.

        By default, this is a meeting-level cleanup and will only run when there is
        no active websocket connection for the meeting. Use `force=True` to override.
        """
        if not force and meeting_id in cls._connections:
            logger.info(
                "Skip clearing session: %s room_name=%s (active websocket connection present)",
                meeting_id,
                room_name,
            )
            return

        session = cls._sessions.pop(meeting_id, None)
        if not session:
            return

        session.participant_histories.clear()
        session.transcript = None
        logger.info(f"Cleared session: {meeting_id} room_name={room_name}")

        prefix = f"{meeting_id}:{room_name or ''}:"
        for key in list(cls._participant_locks.keys()):
            if key.startswith(prefix):
                cls._participant_locks.pop(key, None)