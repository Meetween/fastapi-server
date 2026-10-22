import logging
import asyncio
import time
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
class TranscriptEntry:
    """Single transcript entry with speaker and timestamp"""
    participant_id: Optional[str]
    username: str
    text: str
    timestamp: Optional[float] = None


@dataclass
class Session:
    """
    In-memory session state for a meeting.

    - `meeting_id` / `room_name`: identify the active meeting.
    - `transcript`: structured transcript with speaker + timestamp (List[TranscriptEntry]).
    - `participant_histories`: per-participant chat history (HistoryEntry list).
    - `usermap`: mapping of participant_id <-> username (persistent across join/leave).
    """

    meeting_id: str
    room_name: Optional[str] = None
    transcript: List[TranscriptEntry] = field(default_factory=list)
    participant_histories: Dict[str, List[HistoryEntry]] = field(default_factory=dict)
    usermap: Dict[str, str] = field(default_factory=dict)  # participant_id -> username


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
    # In-memory only: meeting id -> monotonic time Jigasi reported it ended. Pruned by the idle sweep.
    _ended_meetings: Dict[str, float] = {}
    # Monotonic time of the last activity per meeting. Drives the idle sweep.
    _last_activity: Dict[str, float] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance

    # Meeting lifecycle

    @classmethod
    def is_meeting_ended(cls, meeting_id: str) -> bool:
        return meeting_id in cls._ended_meetings

    @classmethod
    async def mark_meeting_ended(cls, meeting_id: str) -> bool:
        """
        Mark a meeting as ended, force-clear its in-memory state, and close its websocket.

        Returns True when the meeting was newly ended, False when it was already ended.
        Clearing is unconditional: an open websocket does not block it.
        """
        newly_ended = meeting_id not in cls._ended_meetings
        if newly_ended:
            cls._ended_meetings[meeting_id] = time.monotonic()

        # Detach the socket before any await so no other task can pick it up.
        websocket = cls._connections.pop(meeting_id, None)
        cls._clear_meeting_state(meeting_id=meeting_id, room_name=None)

        if websocket is not None:
            try:
                await websocket.close(code=1008, reason="meeting ended")
            except Exception as e:
                logger.warning(f"Failed to close websocket for ended meeting {meeting_id}: {e}")

        logger.info(f"Meeting ended: {meeting_id} (newly_ended={newly_ended})")
        return newly_ended

    # Idle sweep

    @classmethod
    def touch_meeting(cls, meeting_id: str) -> None:
        """Record activity for a meeting so the idle clock restarts."""
        cls._last_activity[meeting_id] = time.monotonic()

    @classmethod
    def sweep_idle_meetings(cls, now: Optional[float] = None) -> List[str]:
        """
        Clear meetings with no open websocket and no activity for `meeting_idle_timeout_seconds`.

        Covers meetings whose end signal never arrived. Also prunes expired ended-meeting
        entries and orphaned participant locks. Returns the cleared meeting ids.
        """
        now = time.monotonic() if now is None else now
        cutoff = now - settings.meeting_idle_timeout_seconds
        idle = [
            meeting_id
            for meeting_id, last_seen in cls._last_activity.items()
            if last_seen <= cutoff and meeting_id not in cls._connections
        ]

        for meeting_id in idle:
            session = cls._sessions.get(meeting_id)
            cls._clear_meeting_state(
                meeting_id=meeting_id,
                room_name=session.room_name if session else None,
            )

        cls._prune_ended_meetings(now=now)
        cls._prune_orphaned_locks()

        if idle:
            logger.info(f"Swept {len(idle)} idle meeting(s): {idle}")
        return idle

    @classmethod
    def _prune_ended_meetings(cls, now: float) -> None:
        """Forget ended meetings past `ended_meeting_retention_seconds`. Late updates for them are accepted again after this."""
        cutoff = now - settings.ended_meeting_retention_seconds
        expired = [meeting_id for meeting_id, ended_at in cls._ended_meetings.items() if ended_at <= cutoff]
        for meeting_id in expired:
            del cls._ended_meetings[meeting_id]
        if expired:
            logger.info(f"Pruned {len(expired)} ended meeting(s) past retention")

    @classmethod
    def _prune_orphaned_locks(cls) -> None:
        """Drop unlocked participant locks whose meeting is no longer tracked."""
        for key, lock in list(cls._participant_locks.items()):
            meeting_id = key.split(":", 1)[0]
            if meeting_id not in cls._last_activity and not lock.locked():
                cls._participant_locks.pop(key, None)

    @classmethod
    def _clear_meeting_state(cls, meeting_id: str, room_name: Optional[str]) -> None:
        cls.reset_caption_handler(meeting_id=meeting_id)
        cls.clear_session(meeting_id=meeting_id, room_name=room_name, force=True)

        # A held lock belongs to an in-flight request. Leave it; it is pruned once released.
        lock_prefix = f"{meeting_id}:"
        for key, lock in list(cls._participant_locks.items()):
            if key.startswith(lock_prefix) and not lock.locked():
                cls._participant_locks.pop(key, None)

        cls._last_activity.pop(meeting_id, None)

    # WebSocket connection management

    @classmethod
    def add_connection(cls, meeting_id: str, websocket: WebSocket):
        """Add a WebSocket connection"""
        cls._connections[meeting_id] = websocket
        cls.touch_meeting(meeting_id)
        logger.info(f"Added WebSocket connection: {meeting_id}")

    @classmethod
    def remove_connection(cls, meeting_id: str) -> None:
        """
        Remove a WebSocket connection for a meeting.
        """
        if meeting_id in cls._connections:
            del cls._connections[meeting_id]
            cls.touch_meeting(meeting_id)
            logger.info(f"Removed WebSocket connection: {meeting_id}")

    @classmethod
    def get_connection(cls, meeting_id: str) -> Optional[WebSocket]:
        """Get one WebSocket connection (if any) for a meeting."""
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
        cls.touch_meeting(meeting_id)
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
        cls.touch_meeting(meeting_id)
        session = cls._sessions.get(meeting_id)
        if session is None:
            session = Session(meeting_id=meeting_id, room_name=room_name)
            cls._sessions[meeting_id] = session
            logger.info(f"Created session: {meeting_id}")
        elif room_name and not session.room_name:
            session.room_name = room_name
        return session

    @classmethod
    def update_transcript(
        cls,
        meeting_id: str,
        room_name: Optional[str],
        transcript: Optional[str],
        participant_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Append transcript entry for a meeting.

        Requires participant_id and timestamp to store entry.
        """
        # In-flight websocket messages can outlive the end signal; never recreate state for them.
        if cls.is_meeting_ended(meeting_id):
            logger.info("Dropping transcript for ended meeting: %s", meeting_id)
            return
        session = cls.get_or_create_session(meeting_id=meeting_id, room_name=room_name)
        if not transcript:
            return
        if participant_id is None or timestamp is None:
            logger.warning(
                "Missing attribution for meeting_id=%s: participant_id=%s timestamp=%s",
                meeting_id, participant_id, timestamp,
            )
        username = session.usermap.get(participant_id, "Unknown")
        entry = TranscriptEntry(
            participant_id=participant_id,
            username=username,
            text=transcript,
            timestamp=timestamp,
        )
        session.transcript.append(entry)

        total = 0
        keep_idx = 0
        for i in range(len(session.transcript) - 1, -1, -1):
            e = session.transcript[i]
            total += len(e.username) + len(e.text) + 2
            if total > cls.TRANSCRIPT_CHARS_LIMIT:
                keep_idx = i + 1
                break
        if keep_idx > 0:
            logger.info(
                f"Truncating transcript for meeting_id={meeting_id}: dropping {keep_idx} oldest entries to fit {cls.TRANSCRIPT_CHARS_LIMIT} char budget"
            )
            session.transcript = session.transcript[keep_idx:]

    @classmethod
    def get_transcript(cls, meeting_id: str, room_name: Optional[str]) -> Optional[str]:
        """
        Return transcript formatted as dialogue for LLM context.

        Format: "John Doe: Hello everyone\nJane Smith: Hi John\n..."
        Returns None if no transcript entries exist.
        Read-only: never creates a session, so reads cannot revive an ended or swept meeting.
        """
        session = cls._sessions.get(meeting_id)

        if not session or not session.transcript:
            return None

        dialogue_lines = []
        for entry in session.transcript:
            dialogue_lines.append(f"{entry.username}: {entry.text}")
        return "\n".join(dialogue_lines)

    @classmethod
    def get_participant_history(cls, meeting_id: str, room_name: Optional[str], participant_id: str) -> List[HistoryEntry]:
        """
        Get the in-memory chat history for a participant in a meeting.

        Returns an empty list if no history is stored yet.
        Read-only: never creates a session.
        """
        session = cls._sessions.get(meeting_id)
        if not session:
            return []
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
            logger.info(f"Truncating chat history for meeting_id={meeting_id} participant_id={participant_id} from {len(entries)} to {max_entries}")
            session.participant_histories[participant_id] = entries[-max_entries:]

        logger.info(
            "Stored chat turn for meeting_id=%s participant_id=%s (participant_histories=%d)",
            meeting_id,
            participant_id,
            len(session.participant_histories[participant_id]),
        )

    @classmethod
    def add_participants(
        cls,
        meeting_id: str,
        room_name: Optional[str],
        participants: List[Dict[str, str]],
    ) -> None:
        """
        Add or update participants in usermap.
        Format: [{"participant_id": "xxx", "username": "John Doe"}, ...]
        Participants are not removed.
        """
        session = cls.get_or_create_session(meeting_id=meeting_id, room_name=room_name)

        for p in participants:
            participant_id = p.get("participant_id")
            username = p.get("username")
            if participant_id and username:
                session.usermap[participant_id] = username

        logger.info(
            f"Updated usermap for meeting_id={meeting_id}: {len(session.usermap)} participants"
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

        # Held locks belong to in-flight requests; leave them for _prune_orphaned_locks.
        prefix = f"{meeting_id}:{room_name or ''}:"
        for key, lock in list(cls._participant_locks.items()):
            if key.startswith(prefix) and not lock.locked():
                cls._participant_locks.pop(key, None)
