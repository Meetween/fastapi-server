"""
Chat (Q&A) endpoint. Uses server-side transcript and per-participant history only.

The request `context` field is not used (frontend does not rely on it); meeting text
comes from SessionManager (caption WebSocket transcript).
"""
import logging

from fastapi import APIRouter, HTTPException, Body

from core import EngineManager, SessionManager
from models import (
    HistoryEntry,
    LMMRequest,
    MeetweenAudioRequest,
    MeetweenTaskResponse,
    MeetweenTaskRequest,
)
from utils.audio import uploaded_audio_to_tensor
from utils.prompt_template import format_template

router = APIRouter(prefix="/api", tags=["chat"])
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
async def chat(request: LMMRequest = Body(...)):
    """
    Question and answer endpoint for a meeting participant.

    - Uses server-side transcript built from the caption WebSocket as `context`.
    - Uses server-side per-participant history stored in `SessionManager`.
    - Spoken questions: JSON body with base64 audio (`audio` field), decoded like /api/transcribe.
    """
    engine = EngineManager.get_lmm()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="SpeechLMM model not loaded",
        )
    try:
        audio_tensor = None
        sample_rate = 16000
        if isinstance(request, MeetweenAudioRequest):
            try:
                audio_tensor, sample_rate = uploaded_audio_to_tensor(
                    request.get_audio_bytes()
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            logger.info("Decoded chat audio for generation at %sHz", sample_rate)

        if not isinstance(request, MeetweenTaskRequest):
            raise HTTPException(status_code=400, detail="Invalid request type for chat.")
        task_request = request

        prompt = (task_request.prompt or "").strip()
        if not prompt and audio_tensor is None:
            raise HTTPException(
                status_code=400,
                detail="Field 'prompt' or 'audio' is required for chat.",
            )
        if not prompt and audio_tensor is not None:
            prompt = "(spoken question in attached audio)"

        participant_id = task_request.participant_id or "default"
        async with SessionManager.participant_history_mutex(
            meeting_id=task_request.meeting_id,
            room_name=task_request.room_name,
            participant_id=participant_id,
        ):
            meeting_transcript = SessionManager.get_transcript(
                meeting_id=task_request.meeting_id,
                room_name=task_request.room_name,
            )
            history = SessionManager.get_participant_history(
                meeting_id=task_request.meeting_id,
                room_name=task_request.room_name,
                participant_id=participant_id,
            )
            history_snapshot = list(history)

        combined_prompt = _build_chat_prompt(
            query=prompt,
            context=meeting_transcript,
            history=history_snapshot,
            max_context_chars=engine.max_context_chars,
            max_history_chars=engine.max_history_chars,
        )

        result = await engine._generate_text(
            prompt=combined_prompt,
            audio=audio_tensor,
            sample_rate=sample_rate,
            task="chat",
            max_new_tokens=task_request.max_new_tokens,
            do_sample=task_request.do_sample,
            temperature=task_request.temperature,
            repetition_penalty=task_request.repetition_penalty,
        )

        async with SessionManager.participant_history_mutex(
            meeting_id=task_request.meeting_id,
            room_name=task_request.room_name,
            participant_id=participant_id,
        ):
            SessionManager.store_chat_turn(
                meeting_id=task_request.meeting_id,
                room_name=task_request.room_name,
                participant_id=participant_id,
                user_prompt=prompt,
                assistant_reply=result.strip(),
            )
        return MeetweenTaskResponse(
            type=task_request.type,
            resultType="text",
            room_name=task_request.room_name,
            meeting_id=task_request.meeting_id,
            result=result.strip(),
            language=task_request.language,
            participant_id=task_request.participant_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}",
        )
