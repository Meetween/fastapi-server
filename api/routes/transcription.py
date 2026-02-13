import json
import logging
import tempfile
import os
import time

import numpy as np
from pydub import AudioSegment
from fastapi import APIRouter, HTTPException, Body

from core import EngineManager
from utils import StorageManager, UUIDHelper
from config import settings
from models import (
    MeetweenTranscriptionResponse,
    MeetweenAudioRequest,
    MeetweenTextRequest,
    MeetweenTaskRequest,
    LMMRequest
)
from butler import transcribe_audio

router = APIRouter(prefix="/api", tags=["transcription"])
logger = logging.getLogger(__name__)


@router.post("/transcribe", response_model=MeetweenTranscriptionResponse)
async def api_transcribe(request: LMMRequest = Body(...)):
    """
    Supports three input types:
    - audio: Base64-encoded audio data (MeetweenAudioRequest)
    - text: Plain text input (MeetweenTextRequest)
    - Neither: Basic request (MeetweenTaskRequest), use local recording
    
    Transcribe audio file and return the full transcription
    """
    if not EngineManager.get_lmm():
        raise HTTPException(
            status_code=503,
            detail="SpeechLMM model not loaded"
        )

    try:
        # Handle audio input
        if isinstance(request, MeetweenAudioRequest):
            audio_bytes = request.get_audio_bytes()

            if len(audio_bytes) == 0:
                raise HTTPException(status_code=400, detail="Empty audio data")

            logger.info(f"Transcribing {len(audio_bytes)} bytes, language: {request.language}")

            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            try:
                # Load audio with pydub (better ffmpeg support)
                audio_segment = AudioSegment.from_file(temp_path)

                # Convert to mono, 16kHz (model's native sample rate)
                audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)

                # Get raw PCM data
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
                audio_pcm = audio_array.tobytes()

                logger.info(f"Decoded audio: {len(audio_pcm)} bytes at 16000Hz")

                # Perform transcription
                result = await transcribe_audio(
                    engine=EngineManager.get_lmm(),
                    audio_bytes=audio_pcm,
                    sample_rate=16000,
                    language=request.language or "en-US",
                    participant_id=request.participant_id
                )
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

            return MeetweenTranscriptionResponse(
                type="transcription",
                meeting_id=request.meeting_id,
                participant_id=request.participant_id,
                transcript=result.text,
                message_id=str(UUIDHelper.uuid_from_int(result.message_id)),
                timestamp=result.timestamp,
                is_interim=result.is_interim,
                language=result.language,
                stability=result.stability
            )

        # Handle text input
        elif isinstance(request, MeetweenTextRequest):
            # TODO: Implement text transcription if needed
            raise HTTPException(
                status_code=400,
                detail="Text transcription not yet implemented. Please provide audio data."
            )

        # Handle generic request - load from recordings
        elif isinstance(request, MeetweenTaskRequest):
            meeting_id = request.meeting_id
            room_name = request.room_name
            meeting_path = settings.recordings_base_path / room_name / meeting_id

            if not meeting_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"No recording found for meeting: {room_name}/{meeting_id}"
                )

            # Load metadata
            metadata_path = meeting_path / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            chunk_count = metadata["chunk_count"]

            if chunk_count == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"No audio recorded yet. The meeting just started. No transcription generated"
                )

            logger.info(f"Transcribing meeting {meeting_id}: {chunk_count} chunks, {metadata['total_duration']:.1f}s")

            # Combine all chunks into one audio file
            combined_audio = AudioSegment.empty()
            successful_chunks = 0

            for chunk_num in range(1, chunk_count + 1):
                chunk_path = meeting_path / StorageManager.get_chunk_filename(chunk_num, suffix=settings.recording_format)
                if chunk_path.exists():
                    try:
                        # Verify chunk has content and is not empty
                        if chunk_path.stat().st_size < 100:  # Skip tiny/empty files
                            logger.warning(f"Skipping empty chunk: {chunk_path.name}")
                            continue

                        chunk_audio = AudioSegment.from_file(str(chunk_path))

                        # Skip very short chunks (< 0.5 seconds)
                        if len(chunk_audio) < 500:
                            logger.warning(f"Skipping short chunk: {chunk_path.name} ({len(chunk_audio)}ms)")
                            continue

                        combined_audio += chunk_audio
                        successful_chunks += 1

                    except Exception as e:
                        # Skip corrupted chunks and continue
                        logger.warning(f"Skipping corrupted chunk {chunk_path.name}: {e}")
                        continue

            if len(combined_audio) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"No valid audio found. Recording contains {chunk_count} chunks but none could be decoded."
                )

            logger.info(f"Successfully combined {successful_chunks}/{chunk_count} chunks")

            # Convert to mono, 16kHz
            combined_audio = combined_audio.set_channels(1).set_frame_rate(16000)

            # Get raw PCM data
            audio_array = np.array(combined_audio.get_array_of_samples(), dtype=np.int16)
            audio_pcm = audio_array.tobytes()

            duration_seconds = len(combined_audio) / 1000.0
            logger.info(f"Combined audio: {len(audio_pcm)} bytes, {duration_seconds:.1f}s at 16kHz")

            # Perform transcription
            result = await transcribe_audio(
                engine=EngineManager.get_lmm(),
                audio_bytes=audio_pcm,
                sample_rate=16000,
                language=request.language or "en-US",
                participant_id=request.participant_id
            )

            return MeetweenTranscriptionResponse(
                type="transcription",
                meeting_id=meeting_id,
                participant_id=request.participant_id,
                transcript=result.text,
                message_id=str(UUIDHelper.uuid_from_int(result.message_id)),
                timestamp=result.timestamp,
                is_interim=result.is_interim,
                language=result.language,
                stability=result.stability
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
