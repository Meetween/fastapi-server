"""
Meetween FastAPI Server
Provides SpeechLMM-based API endpoints for transcription, captions, and summarization
Supports WebSocket streaming for real-time transcription
"""
import logging
import uvicorn

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import settings
from core import EngineManager, SessionManager
from src.database import async_mongo_manager
from utils.storage import StorageManager

from api.routes import (
    health,
    caption,
    transcription,
    summary,
    app_config,
    chat,
    agenda,
    auth,
)

from fastapi import FastAPI


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="SpeechLMM-powered transcription, captioning, and summarization API"
)

# Scheduler for periodic tasks
scheduler = AsyncIOScheduler()


def cleanup_old_recordings_job():
    """Scheduled job to delete old recordings"""
    try:
        result = StorageManager.cleanup_old_recordings()
        logger.info(f"Recording cleanup job completed: {result['deleted_count']} recordings deleted")
        if result['errors']:
            logger.warning(f"Cleanup job encountered {len(result['errors'])} errors")
    except Exception as e:
        logger.error(f"Error in recording cleanup job: {str(e)}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    """Initialize server and load SpeechLMM model"""

    logger.info("Server starting up...")

    EngineManager.initialize()
    await async_mongo_manager.connect()
    settings.recordings_base_path.mkdir(parents=True, exist_ok=True)

    # Schedule daily cleanup of old recordings
    scheduler.add_job(
        cleanup_old_recordings_job,
        trigger=CronTrigger(hour=2, minute=0),  # Run daily at 2:00 AM UTC
        id="cleanup_old_recordings",
        name="Cleanup old recordings",
        replace_existing=True
    )
    scheduler.start()
    logger.info("Scheduled recording cleanup job (daily at 2:00 AM UTC)")

    logger.info("Server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    logger.info("Server shutting down...")

    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler shut down")

    await SessionManager.cleanup()
    await EngineManager.cleanup()
    await async_mongo_manager.close()

    logger.info("Server shutdown complete")


app.include_router(health.router)
app.include_router(caption.ws_router)
app.include_router(transcription.router)
app.include_router(summary.router)
app.include_router(app_config.router)
app.include_router(chat.router)
app.include_router(agenda.router)
app.include_router(auth.public_router)
app.include_router(auth.protected_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False
    )
