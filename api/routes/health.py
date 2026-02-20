from fastapi import APIRouter
from core import EngineManager
from config import settings

router = APIRouter(tags=["health"])

@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "speechlmm_loaded": EngineManager.get_lmm() is not None,
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "speechlmm": "loaded" if EngineManager.get_lmm() else "not loaded",
    }
