"""
Chat (Q&A) endpoint. Accepts text prompt, optional context and full chat history;
returns text response.
"""
import logging

from fastapi import APIRouter, HTTPException, Body

from core import EngineManager, SessionManager
from models import MeetweenTaskRequest, MeetweenTaskResponse

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=MeetweenTaskResponse)
async def chat(request: MeetweenTaskRequest = Body(...)):
    """
    Question and answer endpoint for a meeting participant.

    - Uses server-side transcript built from the caption WebSocket as `context`.
    - Uses server-side per-participant history stored in `SessionManager`.
    """
    engine = EngineManager.get_lmm()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="SpeechLMM model not loaded",
        )
    try:
        if not request.prompt:
            raise HTTPException(
                status_code=400,
                detail="Field 'prompt' is required for chat.",
            )

        participant_id = request.participant_id or "default"
        async with SessionManager.participant_history_mutex(
            meeting_id=request.meeting_id,
            room_name=request.room_name,
            participant_id=participant_id,
        ):
            transcript = SessionManager.get_transcript(
                meeting_id=request.meeting_id,
                room_name=request.room_name,
            )
            history = SessionManager.get_participant_history(
                meeting_id=request.meeting_id,
                room_name=request.room_name,
                participant_id=participant_id,
            )

            result = await engine.generate_text_from_text(
                prompt=request.prompt,
                context=transcript,
                history=history,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.do_sample,
                temperature=request.temperature,
                repetition_penalty=request.repetition_penalty
            )

            SessionManager.store_chat_turn(
                meeting_id=request.meeting_id,
                room_name=request.room_name,
                participant_id=participant_id,
                user_prompt=request.prompt,
                assistant_reply=result.strip(),
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
