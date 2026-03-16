"""
Chat (Q&A) endpoint. Accepts text prompt, optional context and full chat history;
returns text response.
"""
import logging

from fastapi import APIRouter, HTTPException, Body

from core import EngineManager
from models import MeetweenTaskRequest, MeetweenTaskResponse

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=MeetweenTaskResponse)
async def chat(request: MeetweenTaskRequest = Body(...)):
    """
    Question and answer endpoint. Send a text prompt with optional context and
    full chat history (e.g. prior Q&A or meeting transcription). Returns text
    response from TritonClient.
    """
    engine = EngineManager.get_lmm()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="SpeechLMM model not loaded",
        )
    try:
        result = await engine.generate_text_from_text(
            prompt=request.prompt,
            context=request.context,
            history=request.history,
            max_new_tokens=512,
            do_sample=False,
        )
        return MeetweenTaskResponse(
            type=request.type,
            resultType="text",
            room_name=request.room_name,
            meeting_id=request.meeting_id,
            result=result.strip(),
            language=request.language,
            participant_id=request.participant_id,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}",
        )
