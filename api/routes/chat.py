"""
Chat (Q&A) endpoint. Uses server-side transcript and per-participant history only.

The request `context` field is not used (frontend does not rely on it); meeting text
comes from SessionManager (caption WebSocket transcript).
"""
import logging
import re

from fastapi import APIRouter, HTTPException, Body

from api.deps.auth import REQUIRE_AUTH
from api.deps.meeting_ended import reject_if_meeting_ended

from core import EngineManager, SessionManager
from models import MeetweenTaskRequest, MeetweenTaskResponse, HistoryEntry, TaskType
from utils.prompt_template import format_template

router = APIRouter(
    prefix="/api",
    tags=["chat"],
    dependencies=REQUIRE_AUTH,
)
logger = logging.getLogger(__name__)

# Value of `error` on a 200 response whose `result` is empty because the meeting has no transcript yet.
NO_TRANSCRIPT_ERROR = "no_transcript"


# Participant ids of the virtual participant start with this prefix. Its answers are read aloud.
VP_PARTICIPANT_PREFIX = "vp:"

SPOKEN_STYLE = (
    "\n- Your answer will be read aloud. Reply in at most two short sentences of plain\n"
    "  spoken English, with no bullet points, headings, markdown or emoji."
    "\n- Answer only from lines of the meeting transcript."
    "\n- If no line of the transcript states the answer, reply exactly: That was not mentioned in the meeting."
    "\n- Never guess. Never give a name, number, date or time that is not written in a transcript line."
    "\n- A fact about a different subject does not answer the question."
    "\n- When asked who said something, name the speaker of that line."
    "\n\nExample, with a different meeting:\n"
    "Zed: The van leaves at noon.\nKim: I will pack the boxes.\n"
    "Question: When does the van leave? Answer: The van leaves at noon.\n"
    "Question: Who drives the van? Answer: That was not mentioned in the meeting.\n"
    "Question: How many boxes are there? Answer: That was not mentioned in the meeting."
)

NOT_MENTIONED = "That was not mentioned in the meeting."

# Words that exist only in the example meeting of SPOKEN_STYLE. The model sometimes copies the example answer.
_EXAMPLE_ONLY_WORDS = re.compile(r"\b(zed|kim|van|boxes|noon|drives)\b", re.IGNORECASE)


def _guard_example_leak(answer: str, transcript: str) -> str:
    """Replace a virtual participant answer that repeats the example meeting instead of the real one."""
    leaked = {m.group(0).lower() for m in _EXAMPLE_ONLY_WORDS.finditer(answer)}
    in_transcript = {m.group(0).lower() for m in _EXAMPLE_ONLY_WORDS.finditer(transcript)}
    if leaked - in_transcript:
        logger.debug("Chat answer repeats the prompt example, replaced: %r", answer)
        return NOT_MENTIONED
    return answer


_LEADING_LABEL = re.compile(r"^\s*([^:\n]{1,40}):\s+")


def _strip_speaker_prefix(answer: str, transcript: str) -> str:
    """Drop a leading "Name:" copied from a transcript line, so the bot does not read the label aloud."""
    match = _LEADING_LABEL.match(answer)
    if not match:
        return answer
    label = match.group(1).strip().lower()
    speakers = {line.split(":", 1)[0].strip().lower() for line in transcript.splitlines() if ":" in line}
    if label not in speakers:
        return answer
    rest = answer[match.end():]
    return rest[:1].upper() + rest[1:]


def _build_chat_prompt(
    query: str,
    context: str | None,
    history: list[HistoryEntry] | None,
    max_context_chars: int,
    max_history_chars: int,
    participant_id: str | None = None,
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
        spoken_style=SPOKEN_STYLE if (participant_id or "").startswith(VP_PARTICIPANT_PREFIX) else "",
    )


@router.post("/chat", response_model=MeetweenTaskResponse)
async def chat(request: MeetweenTaskRequest = Body(...)):
    """
    Question and answer endpoint for a meeting participant.

    - Uses server-side transcript built from the caption WebSocket as `context`.
    - Uses server-side per-participant history stored in `SessionManager`.
    - With no transcript for the meeting the model is not called: the response is 200 with an
      empty `result` and `error` set to `no_transcript`.
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
            # Checked under the history mutex: a turn in flight when the meeting ends must not write back.
            reject_if_meeting_ended(request.meeting_id)
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

        if not (transcript or "").strip():
            # SQA answers questions about the meeting. With no transcript the model can only deflect or
            # invent, so skip it and return an empty result with a reason. The turn is not stored.
            logger.info(
                "Chat skipped, no transcript for meeting_id=%s participant_id=%s",
                request.meeting_id,
                participant_id,
            )
            return MeetweenTaskResponse(
                type=request.type,
                resultType="text",
                room_name=request.room_name,
                meeting_id=request.meeting_id,
                result="",
                language=request.language,
                participant_id=request.participant_id,
                error=NO_TRANSCRIPT_ERROR,
            )

        combined_prompt = _build_chat_prompt(
            query=request.prompt,
            context=transcript,
            history=history_snapshot,
            max_context_chars=engine.max_context_chars,
            max_history_chars=engine.max_history_chars,
            participant_id=request.participant_id,
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

        if participant_id.startswith(VP_PARTICIPANT_PREFIX):
            result = _guard_example_leak(_strip_speaker_prefix(result, transcript), transcript)

        async with SessionManager.participant_history_mutex(
            meeting_id=request.meeting_id,
            room_name=request.room_name,
            participant_id=participant_id,
        ):
            reject_if_meeting_ended(request.meeting_id)
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
