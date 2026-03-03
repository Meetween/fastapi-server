"""
Chat (Q&A) endpoint. Accepts text prompt, optional context and full chat history;
returns text response.
"""
import logging

from fastapi import APIRouter, HTTPException, Body

from core import EngineManager
from models import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest = Body(...)):
    """
    Question and answer endpoint. Send a text prompt with optional context and
    full chat history (e.g. prior Q&A or meeting transcription). Returns text
    response from TritonClient.
    """
    if not EngineManager.get_lmm():
        raise HTTPException(
            status_code=503,
            detail="SpeechLMM model not loaded",
        )

    engine = EngineManager.get_lmm()
    try:
        result = await engine.generate_text_from_text(
            prompt=request.prompt,
            context=request.context,
            history=request.history,
            max_new_tokens=512,
            do_sample=False,
        )
        return ChatResponse(
            result=result.strip(),
            room_name=request.room_name,
            meeting_id=request.meeting_id,
            user_id=request.user_id,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}",
        )
