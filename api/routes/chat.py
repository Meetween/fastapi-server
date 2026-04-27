"""
Chat (Q&A) endpoint. Accepts text prompt, optional context and full chat history;
returns text response.
"""
import logging

from fastapi import APIRouter, HTTPException, Body

from core import EngineManager, SessionManager
from models import MeetweenTaskRequest, MeetweenTaskResponse, HistoryEntry
from utils.shared_prompts import PromptBundle

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)


def _build_chat_prompt(
    query: str,
    context: str | None,
    history: list[HistoryEntry] | None,
    max_context_chars: int,
    max_history_chars: int,
    language: str | None,
) -> str:
    meeting_transcript = (context or "").strip()
    if len(meeting_transcript) > max_context_chars:
        meeting_transcript = meeting_transcript[-max_context_chars:]
        meeting_transcript = "(context truncated for length)\n" + meeting_transcript
    chat_history = ""
    if history:
        chat_history = "\n".join(
            f"{entry.role.capitalize()}: {entry.content}" for entry in history
        ).strip()
        if len(chat_history) > max_history_chars:
            chat_history = chat_history[-max_history_chars:]
            chat_history = "(history truncated for length)\n" + chat_history
    return PromptBundle(language=language or "en-US").chat(
        user_question=query.strip(),
        meeting_transcript=meeting_transcript,
        chat_history=chat_history,
    )


@router.post("/chat", response_model=MeetweenTaskResponse)
async def chat(request: MeetweenTaskRequest = Body(...)):
    """
    Question and answer endpoint for a meeting participant.

    - Uses server-side transcript from the caption WebSocket when present; otherwise
      falls back to the request `context` field (for clients without WebSocket captions).
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
            history_snapshot = list(history)

        meeting_context = (transcript or "").strip()
        if not meeting_context and request.context:
            meeting_context = (request.context or "").strip()

        combined_prompt = _build_chat_prompt(
            query=request.prompt,
            context=meeting_context,
            history=history_snapshot,
            max_context_chars=engine.max_context_chars,
            max_history_chars=engine.max_history_chars,
            language=request.language,
        )
        
        result = await engine._generate_text(
            prompt=combined_prompt,
            audio=None,
            sample_rate=16000,
            task="chat",
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty
        )

        async with SessionManager.participant_history_mutex(
            meeting_id=request.meeting_id,
            room_name=request.room_name,
            participant_id=participant_id,
        ):
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}",
        )
