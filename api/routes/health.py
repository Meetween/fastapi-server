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
        "status": "running",
        "engine_loaded": engine_loaded,
        "speechlmm_loaded": engine_loaded,
        "model_profile": settings.model_profile,
        "triton_url": settings.triton.url,
        "triton_model_name": settings.triton.model_name,
        "prompt_protocol": settings.triton.prompt_protocol,
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    engine_loaded = EngineManager.get_lmm() is not None
    return {
        "status": "healthy",
        "engine": "loaded" if engine_loaded else "not loaded",
        "speechlmm": "loaded" if engine_loaded else "not loaded",
        "model_profile": settings.model_profile,
        "triton_url": settings.triton.url,
        "triton_model_name": settings.triton.model_name,
    }
