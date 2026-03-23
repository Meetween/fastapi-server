"""
Agenda tracking endpoint.
Infers agenda progress from the full meeting transcript.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, get_args

from fastapi import APIRouter, Body, HTTPException

from core import EngineManager, SessionManager
from models import (
    AgendaItem,
    agendaStatusTypes,
    MeetweenAgendaTrackingRequest,
    MeetweenAgendaTrackingResponse,
)
from utils.prompt_template import format_template

router = APIRouter(prefix="/api", tags=["agenda-tracking"])
logger = logging.getLogger(__name__)

_VALID_STATUSES = set(get_args(agendaStatusTypes))


def _extract_json_block(raw_text: str) -> str:
    """Extract the first JSON object/array from model output."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Model output is empty")

    if text.startswith("{") or text.startswith("["):
        return text

    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if not match:
        raise ValueError("No JSON found in model output")
    return match.group(1)


def _extract_status_pairs(raw_text: str) -> List[Dict[str, str]]:
    """
    Extract loose index/status pairs from plain text.
    Examples:
    - "index: 1 status: completed"
    - "index=1, status=in_progress"
    """
    text = (raw_text or "").strip()
    if not text:
        return []

    pattern = re.compile(
        r"index\s*[:=]\s*(\d+)[^\n\r]*?status\s*[:=]\s*([a-zA-Z_]+)",
        flags=re.IGNORECASE,
    )
    return [
        {"index": int(match.group(1)), "status": match.group(2).strip().lower()}
        for match in pattern.finditer(text)
    ]


def _build_items_from_inference(
    inferred_payload: object,
    original_items: List[AgendaItem],
) -> List[AgendaItem]:
    """
    Normalize model output and preserve original ordering.
    Supports:
    - [{"index": 0, "status": "completed"}, ...]
    - {"items": [{"index": 0, "status": "completed"}, ...]}
    - [{"agenda_item": "...", "status": "completed"}, ...]
    """
    inferred_items = inferred_payload
    if isinstance(inferred_payload, dict):
        inferred_items = inferred_payload.get("items")

    if not isinstance(inferred_items, list):
        raise ValueError("Inferred payload must contain an items list")

    by_index: Dict[int, str] = {}
    by_text: Dict[str, str] = {}

    for item in inferred_items:
        if not isinstance(item, dict):
            continue

        status = str(item.get("status", "")).strip().lower()
        if status not in _VALID_STATUSES:
            continue

        idx = item.get("index")
        if isinstance(idx, int):
            by_index[idx] = status
            continue

        text = item.get("text") or item.get("agenda_item")
        if isinstance(text, str) and text.strip():
            by_text[text.strip().lower()] = status

    normalized_items: List[AgendaItem] = []
    for item in original_items:
        status = by_index.get(item.index)
        if status is None:
            status = by_text.get(item.text.strip().lower(), item.status)
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


def _parse_inference_payload(result: str) -> object | None:
    try:
        return json.loads(_extract_json_block(result))
    except ValueError:
        loose_items = _extract_status_pairs(result)
        if loose_items:
            return {"items": loose_items}

    return None


def _build_agenda_prompt(items: List[AgendaItem], transcript: str) -> str:
    agenda_lines = "\n".join(
        f"- index={item.index}, item={item.text}" for item in items
    )
    allowed_statuses = ", ".join(sorted(_VALID_STATUSES))
    return format_template(
        "agenda_prompt",
        filename=str(Path(__file__).resolve().parents[2] / "butler" / "prompts" / "agenda_system_prompt.yml"),
        allowed_statuses=allowed_statuses,
        meeting_transcript=transcript.strip(),
        agenda_items=agenda_lines,
    )


@router.post("/agenda-tracking", response_model=MeetweenAgendaTrackingResponse)
async def agenda_tracking(request: MeetweenAgendaTrackingRequest = Body(...)):
    engine = EngineManager.get_lmm()
    if not engine:
        raise HTTPException(status_code=503, detail="SpeechLMM model not loaded")

    transcript = SessionManager.get_transcript(
        meeting_id=request.meeting_id,
        room_name=request.room_id,
    ) or ""

    if not transcript.strip():
        return _build_agenda_response(request, request.items)
    try:
        agenda_prompt = _build_agenda_prompt(request.items, transcript)
        result = await engine.generate_text_from_text(prompt=agenda_prompt)

        parsed = _parse_inference_payload(result)
        if parsed is None:
            logger.warning(
                "Agenda tracking output not parseable; preserving existing statuses. output=%r",
                (result or "")[:300],
            )
            return _build_agenda_response(request, request.items)

        normalized_items = _build_items_from_inference(parsed, request.items)
        return _build_agenda_response(request, normalized_items)
    except Exception as e:
        logger.error("Agenda tracking failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agenda tracking failed: {str(e)}",
        )
