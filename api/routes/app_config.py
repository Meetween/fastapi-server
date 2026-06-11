from fastapi import APIRouter

from api.deps.auth import REQUIRE_AUTH
from config import settings

router = APIRouter(
    prefix="/api",
    tags=["config"],
    dependencies=REQUIRE_AUTH,
)


@router.get("/config/recording-retention-days")
async def get_recording_retention_days():
    """Get the recording retention days configuration"""
    return {
        "recording_retention_days": settings.recording_retention_days
    }
