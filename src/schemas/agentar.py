"""Pydantic request and response models of the Agentar contract.

Names match `components/schemas` of the contract openapi.yaml. Field rules are the contract's;
the quota and cap numbers that come from Settings are enforced by the routes, not here.
Routes that only relay the vp-worker bot view reuse `VirtualParticipantView`.
"""
from datetime import datetime
from typing import Annotated, Any, Literal, Optional, Union
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from config import settings
from src.schemas.names import agent_name, bot_display_name, person_name
from src.schemas.virtual_participants import BOT_ID_PATTERN, ROOM_PATTERN

OBJECT_ID_PATTERN = r"^[a-f0-9]{24}$"
QUESTION_ID_PATTERN = r"^q[0-9]{1,2}$"
# Opener of a meeting, in characters: three short spoken sentences.
OPENER_MAX_CHARS = 500

ObjectId = Annotated[str, StringConstraints(pattern=OBJECT_ID_PATTERN)]
RoomName = Annotated[str, StringConstraints(pattern=ROOM_PATTERN)]
BotId = Annotated[str, StringConstraints(pattern=BOT_ID_PATTERN)]
VoiceId = Literal["af_heart", "af_bella", "am_fenrir", "am_michael"]
Language = Literal["en"]
# Unicode letters and marks, no parentheses, not the word "agentar". See src/schemas/names.py.
AgentName = Annotated[str, AfterValidator(agent_name)]
PersonName = Annotated[str, StringConstraints(min_length=1, max_length=64), AfterValidator(person_name)]

ErrorCode = Literal[
    "signup_disabled",
    "email_taken",
    "rate_limited",
    "not_found",
    "invalid_state",
    "live_agentar_exists",
    "file_in_use",
    "file_too_large",
    "unsupported_type",
    "picture_invalid",
    "quota_files",
    "quota_bytes",
    "quota_meetings_per_day",
    "room_invalid",
    "start_in_past",
    "report_not_ready",
    "jitsi_key_missing",
    "worker_unavailable",
    "storage_unavailable",
    "agentar_capacity",
    "room_not_owned",
    "length_required",
    "event_clock_skew",
    "model_unavailable",
]


class Error(BaseModel):
    """Body of every error of the new routes. `detail` stays a string for current clients."""

    detail: str
    code: ErrorCode


class PlainError(BaseModel):
    """Body of the existing routes' HTTPException."""

    detail: str


# Profile ------------------------------------------------------------------------------------


class Profile(BaseModel):
    agent_name: Optional[str] = None
    voice: Optional[VoiceId] = None
    language: Language = "en"
    onboarded_at: Optional[datetime] = None
    onboarding_skipped_at: Optional[datetime] = None
    has_picture: bool = False


class Me(BaseModel):
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_admin: bool
    profile: Profile
    onboarding_state: Literal["pending", "done", "skipped"]


class MeProfilePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: Optional[AgentName] = None
    voice: Optional[VoiceId] = None
    language: Optional[Language] = None

    @model_validator(mode="after")
    def _not_empty(self) -> "MeProfilePatch":
        if not self.model_fields_set:
            raise ValueError("profile needs at least one field")
        return self


class MePatch(BaseModel):
    """Partial update. Only fields in `model_fields_set` are written, so an explicit
    `agent_name: null` clears the name and an omitted one leaves it."""

    model_config = ConfigDict(extra="forbid")

    first_name: Optional[PersonName] = None
    last_name: Optional[PersonName] = None
    profile: Optional[MeProfilePatch] = None
    complete_onboarding: Optional[Literal[True]] = None

    @model_validator(mode="after")
    def _not_empty(self) -> "MePatch":
        if not self.model_fields_set:
            raise ValueError("body needs at least one field")
        return self


class Picture(BaseModel):
    width: Literal[512] = 512
    height: Literal[512] = 512
    bytes: int = Field(..., le=409_600)
    sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    updated_at: datetime


class VoiceInfo(BaseModel):
    id: VoiceId
    label: str
    gender: Literal["female", "male"]
    sample_url: str
    default: bool


# Files --------------------------------------------------------------------------------------

FileScope = Literal["general", "meeting"]
FileContentType = Literal[
    "text/plain",
    "text/markdown",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
]


class AgentarFile(BaseModel):
    """File metadata as the API shows it. Chunks and the storage key never leave the API."""

    id: ObjectId
    scope: FileScope
    meeting_id: Optional[ObjectId] = None
    name: str = Field(..., max_length=255)
    size: int = Field(..., ge=1)
    content_type: FileContentType
    sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    text_status: Literal["ok", "empty", "truncated", "failed"]
    text_chars: int = Field(..., ge=0, le=200_000)
    chunk_count: int = Field(..., ge=0)
    error: Optional[str] = None
    created_at: datetime


class Usage(BaseModel):
    files: int
    files_max: int
    bytes: int
    bytes_max: int
    file_max_bytes: int


class FileList(BaseModel):
    items: list[AgentarFile]
    usage: Usage


# Meetings -----------------------------------------------------------------------------------

MeetingStatus = Literal[
    "draft", "scheduled", "waiting_for_human", "joining", "live", "leaving", "ended", "missed", "failed"
]
MeetingMode = Literal["now", "scheduled"]
ReportStatus = Literal["none", "pending", "ready", "failed"]
QuestionText = Annotated[str, StringConstraints(min_length=1, max_length=300)]


class Question(BaseModel):
    id: str = Field(..., pattern=QUESTION_ID_PATTERN, description="Assigned by position on every write")
    text: str = Field(..., min_length=1, max_length=300)


class MeetingFields(BaseModel):
    """Writable meeting fields. `MeetingCreate` requires `room` and `mode`; `MeetingPatch`
    writes only the fields in `model_fields_set`."""

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(default=None, max_length=120)
    room: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=300,
        description=(
            "A room name or a link on JITSI_MEET_BASE_URL. The server keeps the last path "
            "segment, lowercased, and checks RoomName. Another host answers 422 room_invalid."
        ),
    )
    mode: Optional[MeetingMode] = None
    start_at: Optional[datetime] = Field(
        default=None,
        description="Required for scheduled, 1 minute to 30 days ahead. Ignored for now.",
    )
    grace_minutes: int = Field(default=settings.agentar.grace_minutes_default, ge=1, le=60)
    purpose: str = Field(default="", max_length=1000)
    opener: str = Field(
        default="",
        max_length=OPENER_MAX_CHARS,
        description=(
            "Plain text the Agentar speaks once after the greeting, folded to one line. Empty speaks "
            "the purpose sentence. POST .../opener-draft proposes one."
        ),
    )
    questions: list[QuestionText] = Field(default_factory=list, max_length=10)
    relevant_info: str = Field(default="", max_length=2000)
    file_ids: list[ObjectId] = Field(default_factory=list, max_length=20)
    use_general_kb: bool = False

    @field_validator("file_ids")
    @classmethod
    def _unique_file_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("file_ids must be unique")
        return value


class MeetingCreate(MeetingFields):
    room: str = Field(..., min_length=1, max_length=300)
    mode: MeetingMode


class MeetingPatch(MeetingFields):
    @model_validator(mode="after")
    def _not_empty(self) -> "MeetingPatch":
        if not self.model_fields_set:
            raise ValueError("body needs at least one field")
        return self


class Meeting(BaseModel):
    id: ObjectId
    title: Optional[str] = None
    room_name: RoomName
    join_url: Optional[str] = Field(default=None, description="Plain room URL for humans, no token")
    mode: MeetingMode
    start_at: Optional[datetime] = None
    grace_minutes: int
    purpose: str
    opener: str = ""
    questions: list[Question]
    relevant_info: str
    file_ids: list[ObjectId]
    use_general_kb: bool
    has_picture: bool
    status: MeetingStatus
    status_reason: Optional[str] = Field(
        default=None,
        description=(
            "For missed and failed: no_human_joined, another_agentar_live, agentar_capacity, "
            "room_not_owned, worker_error, storage_unavailable, jitsi_key_missing, join_timeout, bot_lost; "
            "room_stayed_empty on meetings missed before the Agentar opened the room itself"
        ),
    )
    display_name: Optional[str] = None
    voice: Optional[VoiceId] = None
    bot_id: Optional[str] = None
    jitsi_meeting_id: Optional[str] = None
    dispatched_at: Optional[datetime] = None
    joined_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    report_status: ReportStatus
    created_at: datetime
    updated_at: datetime


class OpenerDraft(BaseModel):
    opener: str = Field(
        ...,
        max_length=OPENER_MAX_CHARS,
        description="At most three short sentences, every figure and name taken from the purpose or the files. Not saved.",
    )


class MeetingList(BaseModel):
    items: list[Meeting]
    next_cursor: Optional[str] = None


# Worker events and notifications ------------------------------------------------------------

EventType = Literal["joined", "question_asked", "answer_given", "escalation", "left", "error"]
EventChannel = Literal["group", "private", "voice"]


class _EventData(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EventJoined(_EventData):
    kind: Literal["joined"]
    join_ms: Optional[int] = Field(default=None, ge=0)


class EventQuestionAsked(_EventData):
    kind: Literal["question_asked"]
    question_id: str = Field(..., pattern=QUESTION_ID_PATTERN)
    text: str = Field(..., max_length=300)
    first_word_ms: Optional[int] = Field(default=None, ge=0)


class EventAnswerGiven(_EventData):
    kind: Literal["answer_given"]
    question: str = Field(..., max_length=2048)
    answer: str = Field(..., max_length=2000)
    asker: Optional[str] = Field(..., max_length=64)
    channel: Optional[EventChannel] = None
    first_word_ms: Optional[int] = Field(default=None, ge=0)


class EventEscalation(_EventData):
    kind: Literal["escalation"]
    question: str = Field(..., max_length=2048)
    asker: Optional[str] = Field(..., max_length=64)
    channel: Optional[EventChannel] = None


class EventLeft(_EventData):
    kind: Literal["left"]
    reason: Literal["recalled", "alone", "max_lifetime", "conference_ended", "error"]


class EventError(_EventData):
    kind: Literal["error"]
    message: str = Field(..., max_length=500, description="No secrets, no tokens")
    fatal: bool


EventData = Annotated[
    Union[EventJoined, EventQuestionAsked, EventAnswerGiven, EventEscalation, EventLeft, EventError],
    Field(discriminator="kind"),
]


class WorkerEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: UUID = Field(..., description="Generated by the worker, idempotency key")
    agentar_meeting_id: ObjectId
    bot_id: BotId
    type: EventType
    at: datetime = Field(..., description="Worker clock")
    jitsi_meeting_id: Optional[str] = Field(default=None, max_length=128)
    data: EventData

    @model_validator(mode="after")
    def _type_matches_data(self) -> "WorkerEvent":
        if self.data.kind != self.type:
            raise ValueError("data.kind must equal type")
        return self


class EventAck(BaseModel):
    accepted: Literal[True] = True
    duplicate: bool


NotificationType = Literal[
    "joined", "question_asked", "answer_given", "escalation", "left", "error", "missed", "failed", "report_ready"
]


class Notification(BaseModel):
    id: str = Field(..., description="Opaque, ordered")
    at: datetime = Field(..., description="Server receive time")
    meeting_id: ObjectId
    meeting_title: Optional[str] = None
    type: NotificationType
    severity: Literal["info", "warning", "error"]
    data: dict[str, Any]


class NotificationList(BaseModel):
    items: list[Notification]
    cursor: Optional[str] = Field(default=None, description="Pass as `since` on the next poll")


# Reports ------------------------------------------------------------------------------------


class TranscriptLine(BaseModel):
    n: int = Field(..., ge=1, description="Line number cited by transcript_refs")
    speaker: str
    text: str
    at: Optional[datetime] = None
    source: Literal["speech", "chat", "agentar"]


class ActionItem(BaseModel):
    text: str
    owner: str = Field(..., description='Speaker name or "unassigned"')
    due: Optional[str] = None
    quote: Optional[str] = Field(default=None, description="Verbatim substring of the cited line")
    transcript_refs: list[Annotated[int, Field(ge=1)]] = Field(..., min_length=1)


class Answer(BaseModel):
    question_id: str
    question: str
    status: Literal["answered", "no_answer_captured", "not_asked"]
    answer: Optional[str] = None
    speaker: Optional[str] = None
    quote: Optional[str] = Field(default=None, description="Verbatim substring of the cited lines")
    transcript_refs: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list)


class OpenPoint(BaseModel):
    kind: Literal["escalation", "unanswered_question", "action_for_owner"]
    text: str
    asker: Optional[str] = None
    transcript_refs: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list)


class Report(BaseModel):
    meeting_id: ObjectId
    status: Literal["pending", "ready", "failed"]
    error: Optional[str] = None
    room_name: Optional[RoomName] = None
    title: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    generated_at: Optional[datetime] = None
    generation_ms: Optional[int] = Field(default=None, description="From ended_at to ready; the bar is 60000")
    summary: Optional[str] = None
    action_items: list[ActionItem] = Field(default_factory=list)
    answers: list[Answer] = Field(default_factory=list)
    open_points: list[OpenPoint] = Field(default_factory=list)
    transcript: list[TranscriptLine] = Field(default_factory=list)


# vp-worker ----------------------------------------------------------------------------------


class AgentarWorkerQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=QUESTION_ID_PATTERN)
    text: str = Field(..., min_length=1, max_length=300)


OWNER_FIRST_NAME_CHARS = 32


class AgentarWorkerBot(BaseModel):
    """Mirror of the worker's `CreateBot` (vp-worker-bots.delta.yaml) with `extra="forbid"`.
    Built only by the dispatch code from the meeting and the owner's profile, never from a
    client body. `display_name` is "<Name> Agentar"; the worker appends its suffix."""

    model_config = ConfigDict(extra="forbid")

    room: RoomName
    display_name: Annotated[str, AfterValidator(bot_display_name)] = "Agentar"
    jwt: Optional[str] = Field(default=None, max_length=4096)
    script: Optional[str] = Field(default=None, max_length=2000)
    greeting: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    opener: Optional[str] = Field(default=None, min_length=1, max_length=1000)
    questions: list[AgentarWorkerQuestion] = Field(default_factory=list, max_length=10)
    escalation_line: Optional[str] = Field(default=None, min_length=1, max_length=300)
    farewell: Optional[str] = Field(default=None, min_length=1, max_length=300)
    voice: Optional[VoiceId] = None
    avatar_jpeg_b64: Optional[str] = Field(default=None, max_length=546_136)
    agentar_meeting_id: Optional[ObjectId] = None
    events_url: Optional[str] = Field(default=None, max_length=512)
    # The owner's first name, cleaned to the worker's display name characters; None when the
    # profile has none. A chat message that starts with it asks the bot (owner decision C).
    owner_first_name: Optional[str] = Field(default=None, min_length=1, max_length=OWNER_FIRST_NAME_CHARS)

    @model_validator(mode="after")
    def _script_or_opener(self) -> "AgentarWorkerBot":
        if self.script and self.opener:
            raise ValueError("script and opener are mutually exclusive")
        if self.agentar_meeting_id and not self.events_url:
            raise ValueError("events_url is required with agentar_meeting_id")
        return self
