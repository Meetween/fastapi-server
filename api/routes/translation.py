import io
import logging
from typing import Annotated, Union

from pydub import AudioSegment
from fastapi import APIRouter, HTTPException, Body

from core import EngineManager
from utils import pydub_to_pcm16
from models import (
    MeetweenTranslationResponse,
    MeetweenMTRequest,
    MeetweenSTRequest,
    TaskType,
)
from butler import translate_audio, translate_text

router = APIRouter(prefix="/api", tags=["translation"])

logger = logging.getLogger(__name__)

# Discriminated union so FastAPI correctly deserializes ST vs MT by the `type` field
TranslationRequest = Annotated[
    Union[MeetweenSTRequest, MeetweenMTRequest],
    Body(discriminator="type"),
]


async def handle_speech_request(request: MeetweenSTRequest) -> MeetweenTranslationResponse:
    audio_bytes = request.get_audio_bytes()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio data")

    logger.info(f"Translating {len(audio_bytes)} bytes, {request.language}->{request.target_language}")

    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        audio_pcm = pydub_to_pcm16(audio_segment)
    except Exception as e:
        logger.warning(f"Failed to decode audio: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid or unsupported audio format: {e}")

    logger.info(f"Decoded audio: {len(audio_pcm)} bytes at 16000Hz")

    translation = await translate_audio(
        engine=EngineManager.get_lmm(),
        audio_bytes=audio_pcm,
        language=request.language,
        target_lang=request.target_language,
        sample_rate=16000,
        participant_id=request.participant_id,
    )

    return MeetweenTranslationResponse(
        type=TaskType.SPEECH_TRANSLATION,
        resultType="text",
        room_name=request.room_name,
        meeting_id=request.meeting_id,
        result=translation,
        language=request.language,
        target_language=request.target_language,
        participant_id=request.participant_id,
    )


async def handle_text_request(request: MeetweenMTRequest) -> MeetweenTranslationResponse:
    logger.info(f"Translating text, {request.language}->{request.target_language}")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided for translation")

    translation = await translate_text(
        engine=EngineManager.get_lmm(),
        input_text=request.text,
        language=request.language,
        target_lang=request.target_language,
        participant_id=request.participant_id,
    )

    return MeetweenTranslationResponse(
        type=TaskType.TEXT_TRANSLATION,
        resultType="text",
        room_name=request.room_name,
        meeting_id=request.meeting_id,
        result=translation,
        language=request.language,
        target_language=request.target_language,
        participant_id=request.participant_id,
    )


@router.post("/translate", response_model=MeetweenTranslationResponse)
async def api_translate(request: TranslationRequest):
    """
    Supports two input types:
    - ST (type=TaskType.SPEECH_TRANSLATION): Base64-encoded audio — speech translation (MeetweenSTRequest)
    - MT (type=TaskType.TEXT_TRANSLATION): Plain text — text translation (MeetweenMTRequest)

    Returns a single translation result.
    """
    if not EngineManager.get_lmm():
        raise HTTPException(status_code=503, detail="SpeechLMM model not loaded")

    try:
        if isinstance(request, MeetweenSTRequest):
            return await handle_speech_request(request)
        else:
            return await handle_text_request(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
