from src.database.models.agentar_event import AgentarEvent
from src.database.models.agentar_file import AgentarFile, FileChunk
from src.database.models.agentar_meeting import AgentarMeeting, MeetingQuestion
from src.database.models.agentar_report import AgentarReport
from src.database.models.meeting_context import MeetingContext
from src.database.models.user import User, UserProfile
from src.database.models.vp_bot_owner import VpBotOwner

# Every model with a collection, in index creation order. Startup and the test reset walk this.
ALL_MODELS = (
    User,
    MeetingContext,
    AgentarFile,
    AgentarMeeting,
    AgentarReport,
    AgentarEvent,
    VpBotOwner,
)

__all__ = [
    "ALL_MODELS",
    "AgentarEvent",
    "AgentarFile",
    "AgentarMeeting",
    "AgentarReport",
    "FileChunk",
    "MeetingContext",
    "MeetingQuestion",
    "User",
    "UserProfile",
    "VpBotOwner",
]
