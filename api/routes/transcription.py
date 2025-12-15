import logging
from fastapi import APIRouter, UploadFile, Form, HTTPException, File
from typing import Optional
from models import MeetweenTranscriptionResponse
from butler import transcribe_audio
from core import EngineManager

router = APIRouter(prefix="/api", tags=["transcription"])
logger = logging.getLogger(__name__)


@router.post("/transcribe", response_model=MeetweenTranscriptionResponse)
async def api_transcribe(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form("en-US", description="BCP-47 language code (e.g., 'en-US', 'es-ES', 'fr-FR')"),
    participant_id: Optional[str] = Form(None, description="Optional participant identifier")
):
    """
    Transcribe audio file and return the full transcription

    - **audio**: Audio file (WAV, MP3, etc.)
    - **language**: BCP-47 language code (default: 'en-US')
    - **participant_id**: Optional participant identifier
    """
    if not EngineManager.get_lmm():
        raise HTTPException(
            status_code=503,
            detail="SpeechLMM model not loaded. Check server configuration."
        )

    try:
        # Read audio file
        audio_data = await audio.read()
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        logger.info(f"Transcribing {len(audio_data)} bytes, language: {language}")

        # Perform transcription
        result = transcribe_audio(
            audio_data=audio_data,
            engine=EngineManager.get_lmm(),
            language=language,
            vad_model=EngineManager.get_vad()
        )

        # Build response
        return MeetweenTranscriptionResponse(
            transcript=result.text,
            is_final=True,
            confidence=result.confidence,
            transcript_id=str(uuid.uuid4()),
            participant_id=participant_id,
            language=language,
            message_id=0,
            timestamp=0.0,
            is_interim=False,
            stability=1.0
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
