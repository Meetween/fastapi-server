from fastapi import APIRouter
from core import EngineManager
from config import settings

router = APIRouter(tags=["health"])

@router.get("/")
async def root():
    """Root endpoint"""
    engine_loaded = EngineManager.get_lmm() is not None
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "commit": settings.app_commit,
        "status": "running",
        "engine_loaded": engine_loaded,
        "model_profile": settings.model_profile,
        "triton_url": settings.triton.url,
        "triton_model_name": settings.triton.model_name,
        "prompt_protocol": settings.triton.prompt_protocol,
        "closed_caption_engine": _closed_caption_status(),
    }


def _closed_caption_status() -> dict:
    """Closed-caption engine status. Readiness is keyed to the primary engine only."""
    cc = settings.closed_caption_triton
    if cc is None:
        return {"configured": False, "status": "not configured"}
    if EngineManager.is_closed_caption_loaded():
        return {
            "configured": True,
            "status": "loaded",
            "model_profile": settings.closed_caption_model_profile,
            "triton_url": cc.url,
            "triton_model_name": cc.model_name,
        }
    # Configured but not connected: caption requests fall back to the primary engine.
    return {
        "configured": True,
        "status": "degraded",
        "model_profile": settings.closed_caption_model_profile,
        "triton_url": cc.url,
        "triton_model_name": cc.model_name,
        "fallback": "primary engine",
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    engine_loaded = EngineManager.get_lmm() is not None
    return {
        "status": "healthy",
        "version": settings.app_version,
        "commit": settings.app_commit,
        "engine": "loaded" if engine_loaded else "not loaded",
        "model_profile": settings.model_profile,
        "triton_url": settings.triton.url,
        "triton_model_name": settings.triton.model_name,
        "closed_caption_engine": _closed_caption_status(),
    }
