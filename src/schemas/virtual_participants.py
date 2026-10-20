from pydantic import BaseModel, Field


class VpCreateRequest(BaseModel):
    meeting_id: str = Field(..., min_length=1, description="Meeting identifier")
    room_name: str = Field(..., min_length=1, description="Jitsi room name")
    name: str = Field(..., min_length=1, description="Persona the agent personifies")
    avatar_id: str | None = Field(default=None, description="Avatar to render")
    human_count: int = Field(
        ...,
        ge=0,
        description="Humans currently in the room, as reported by the client",
    )


class VpBotResponse(BaseModel):
    bot_id: str
    room_name: str
    state: str = Field(..., description="'pending' before the worker accepts, then 'active'")
    worker_id: str
