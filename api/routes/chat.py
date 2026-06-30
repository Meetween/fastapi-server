"""
Chat (Q&A) endpoint. Uses server-side transcript and per-participant history only.

The request `context` field is not used (frontend does not rely on it); meeting text
comes from SessionManager (caption WebSocket transcript).
"""
import logging

from fastapi import APIRouter, HTTPException, Body

from api.deps.auth import REQUIRE_AUTH

from core import EngineManager, SessionManager
from models import MeetweenTaskRequest, MeetweenTaskResponse, HistoryEntry, TaskType
from utils.prompt_template import format_template

router = APIRouter(
    prefix="/api",
    tags=["chat"],
    dependencies=REQUIRE_AUTH,
)
logger = logging.getLogger(__name__)


def _build_chat_prompt(
    query: str,
    context: str | None,
    history: list[HistoryEntry] | None,
    max_context_chars: int,
    max_history_chars: int,
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
    return format_template(
        "chat_prompt",
        meeting_transcript=meeting_transcript,
        chat_history=chat_history,
        user_question=query.strip(),
    )


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
        if request.type != TaskType.SPOKEN_QUESTION_ANSWERING:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Invalid chat task type '{request.type}'. "
                    f"Expected '{TaskType.SPOKEN_QUESTION_ANSWERING}'."
                ),
            )
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

        combined_prompt = _build_chat_prompt(
            query=request.prompt,
            context=transcript,
            history=history_snapshot,
            max_context_chars=engine.max_context_chars,
            max_history_chars=engine.max_history_chars,
        )
        
        result = await engine._generate_text(
            prompt=combined_prompt,
            audio=None,
            sample_rate=16000,
            task=TaskType.SPOKEN_QUESTION_ANSWERING,
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
