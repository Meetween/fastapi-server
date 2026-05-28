"""
Agenda tracking endpoint.
Infers agenda progress from the full meeting transcript.
"""
import json
import logging
import re
from typing import Dict, List, get_args

from fastapi import APIRouter, Body, HTTPException, Response

from api.deps.auth import REQUIRE_AUTH
from pydantic import ValidationError

from core import EngineManager, SessionManager
from models import (
    AgendaItem,
    AgendaTrackingModelOutput,
    agendaStatusTypes,
    MeetweenAgendaTrackingRequest,
    MeetweenAgendaTrackingResponse,
)
from utils.prompt_template import format_template

router = APIRouter(
    prefix="/api",
    tags=["agenda-tracking"],
    dependencies=REQUIRE_AUTH,
)
logger = logging.getLogger(__name__)

_VALID_STATUSES = set(get_args(agendaStatusTypes))


def _normalize_agenda_text(text: str) -> str:
    """
    Normalize agenda text for tolerant comparisons.
    Ignores casing, whitespace, and punctuation-only differences.
    """
    return re.sub(r"[\W_]+", "", text.casefold())


def _agenda_items_output_schema() -> dict:
    """JSON Schema for model output: object with an items array of AgendaItem shapes."""
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": AgendaItem.model_json_schema(),
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def _decode_json(raw_text: str) -> object:
    """Strictly decode JSON from the entire LLM response."""
    if not raw_text:
        raise ValueError("Model output is empty")
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in model output: {raw_text}")


def _build_items_from_inference(
    inferred_items: List[AgendaItem],
    original_items: List[AgendaItem],
) -> List[AgendaItem]:
    """
    Normalize model output and preserve original ordering.
    Uses index-based status updates while preserving original ordering and text.
    """
    by_index: Dict[int, str] = {}
    for item in inferred_items:
        by_index[item.index] = item.status

    normalized_items: List[AgendaItem] = []
    for item in original_items:
        status = by_index.get(item.index, item.status)
        normalized_items.append(
            AgendaItem(index=item.index, text=item.text, status=status)
        )

    return normalized_items


def _build_agenda_response(
    request: MeetweenAgendaTrackingRequest,
    items: List[AgendaItem],
) -> MeetweenAgendaTrackingResponse:
    return MeetweenAgendaTrackingResponse(
        meeting_id=request.meeting_id,
        room_id=request.room_id,
        items=items,
    )


def _inference_items_match_request(
    inferred: List[AgendaItem],
    original: List[AgendaItem],
) -> bool:
    """
    Ensure model output matches the request: same indices, no duplicates,
    and each item text matches the corresponding agenda line after normalization.
    """
    if len(inferred) != len(original):
        return False
    orig_by_index: Dict[int, AgendaItem] = {o.index: o for o in original}
    expected_indices = set(orig_by_index.keys())
    seen: set[int] = set()
    for item in inferred:
        if item.index in seen:
            return False
        seen.add(item.index)
        orig = orig_by_index.get(item.index)
        if orig is None:
            return False
        if _normalize_agenda_text(item.text) != _normalize_agenda_text(orig.text):
            return False
    return seen == expected_indices


def _parse_inference_payload(
    result: str,
    original_items: List[AgendaItem],
) -> List[AgendaItem] | None:
    try:
        decoded = _decode_json(result)
        validated = AgendaTrackingModelOutput.model_validate(decoded)
        if not _inference_items_match_request(validated.items, original_items):
            return None
        return list(validated.items)
    except (ValueError, ValidationError):
        return None

def _build_agenda_prompt(items: List[AgendaItem], transcript: str) -> str:
    agenda_lines = "\n".join(
        f"- index={item.index}, item={item.text}" for item in items
    )
    allowed_statuses = ", ".join(sorted(_VALID_STATUSES))
    json_schema = json.dumps(_agenda_items_output_schema(), indent=2)
    return format_template(
        "agenda_prompt",
        filename="agenda_system_prompt.yml",
        allowed_statuses=allowed_statuses,
        meeting_transcript=transcript.strip(),
        agenda_items=agenda_lines,
        json_schema=json_schema,
    )


@router.post(
    "/agenda-tracking",
    response_model=MeetweenAgendaTrackingResponse,
    responses={204: {"description": "No Content"}},
)
async def agenda_tracking(
    request: MeetweenAgendaTrackingRequest = Body(...),
) -> MeetweenAgendaTrackingResponse | Response:
    engine = EngineManager.get_lmm()
    if not engine:
        raise HTTPException(status_code=503, detail="SpeechLMM model not loaded")

    transcript = SessionManager.get_transcript(
        meeting_id=request.meeting_id,
        room_name=request.room_id,
    ) or ""

    if not transcript.strip():
        return Response(status_code=204)
    try:
        agenda_prompt = _build_agenda_prompt(request.items, transcript)
        result = await engine._generate_text(
            prompt=agenda_prompt,
            audio=None,
            sample_rate=16000,
            task="chat",
            do_sample=False,
        )
        parsed = _parse_inference_payload(result, request.items)
        if parsed is None:
            logger.warning(
                "Agenda tracking output invalid (parse/schema or mismatch with request items); "
                "preserving existing statuses. output=%r",
                (result or "")[:300],
            )
            return Response(status_code=204)

        normalized_items = _build_items_from_inference(parsed, request.items)
        return _build_agenda_response(request, normalized_items)
    except Exception as e:
        logger.error("Agenda tracking failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agenda tracking failed: {str(e)}",
        )
