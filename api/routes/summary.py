import json
import logging
import tempfile
import os

import numpy as np
from pydub import AudioSegment
from fastapi import APIRouter, HTTPException, Body

from core import EngineManager
from config import settings
from models import (
    MeetweenTaskResponse,
    MeetweenAudioRequest,
    MeetweenTextRequest,
    MeetweenTaskRequest,
    MeetweenCatchupRequest,
    LMMRequest
)
from butler import summarize_audio

router = APIRouter(prefix="/api", tags=["summary"])

logger = logging.getLogger(__name__)


@router.post("/summarize", response_model=MeetweenTaskResponse)
async def api_summarize(request: LMMRequest = Body(...)):
    """
    Supports three input types:
    - audio: Base64-encoded audio data (MeetweenAudioRequest)
    - text: Plain text input (MeetweenTextRequest)
    - Neither: Basic request (MeetweenTaskRequest), use local recording

    Returns a summary with task metadata
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

            logger.info(f"Summarizing {len(audio_bytes)} bytes, language: {request.language}")

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

                # Generate summary
                summary = summarize_audio(
                    audio_bytes=audio_pcm,
                    sample_rate=16000,
                    engine=EngineManager.get_lmm(),
                    language=request.language
                )
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

            return MeetweenTaskResponse(
                type="summary",
                resultType="text",
                room_name=request.room_name,
                meeting_id=request.meeting_id,
                result=summary,
                language=request.language,
                participant_id=request.participant_id
            )

        # Handle text input
        elif isinstance(request, MeetweenTextRequest):
            # TODO
            return MeetweenTaskResponse(
                type="summary",
                resultType="text",
                room_name=request.room_name,
                meeting_id=request.meeting_id,
                result=f"Text summary requested: Please implement me",
                language=request.language,
                participant_id=request.participant_id
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
                return MeetweenTaskResponse(
                    type="summary",
                    resultType="text",
                    room_name=request.room_name,
                    meeting_id=meeting_id,
                    result="No audio recorded yet. The meeting just started.",
                    language=request.language,
                    participant_id=request.participant_id
                )

            logger.info(f"Summarizing meeting {meeting_id}: {chunk_count} chunks, {metadata['total_duration']:.1f}s")

            # Combine all chunks into one audio file
            combined_audio = AudioSegment.empty()
            successful_chunks = 0

            for chunk_num in range(1, chunk_count + 1):
                chunk_path = meeting_path / f"chunk_{chunk_num:04d}.webm"
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
                return MeetweenTaskResponse(
                    type="summary",
                    resultType="text",
                    room_name=request.room_name,
                    meeting_id=meeting_id,
                    result=f"No valid audio found. Recording contains {chunk_count} chunks but none could be decoded.",
                    language=request.language,
                    participant_id=request.participant_id
                )

            logger.info(f"Successfully combined {successful_chunks}/{chunk_count} chunks")

            # Convert to mono, 16kHz
            combined_audio = combined_audio.set_channels(1).set_frame_rate(16000)

            # Get raw PCM data
            audio_array = np.array(combined_audio.get_array_of_samples(), dtype=np.int16)
            audio_pcm = audio_array.tobytes()

            duration_seconds = len(combined_audio) / 1000.0
            logger.info(f"Combined audio: {len(audio_pcm)} bytes, {duration_seconds:.1f}s at 16kHz")

            # Generate summary
            summary = summarize_audio(
                audio_bytes=audio_pcm,
                sample_rate=16000,
                engine=EngineManager.get_lmm(),
                language=request.language or "en-US"
            )

            return MeetweenTaskResponse(
                type="summary",
                resultType="text",
                room_name=request.room_name,
                meeting_id=meeting_id,
                result=summary,
                language=request.language,
                participant_id=request.participant_id
            )

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid request type. Must provide audio, text, or meeting_id."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@router.post("/catchup", response_model=MeetweenTaskResponse)
async def api_catchup(request: MeetweenCatchupRequest = Body(...)):
    """
    Generate a "catch me up" summary of recorded meeting audio.

    This endpoint loads recorded audio chunks for a meeting and generates a summary.

    - **meeting_id**: Meeting identifier to load recordings from
    - **max_duration**: Optional maximum audio duration to process (in seconds)

    Returns a catch_me_up type response with summary
    """
    if not EngineManager.get_lmm():
        raise HTTPException(
            status_code=503,
            detail="SpeechLMM model not loaded"
        )

    try:
        meeting_id = request.meeting_id
        meeting_path = settings.recordings_base_path / meeting_id

        if not meeting_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No recording found for meeting: {meeting_id}"
            )

        # Load metadata
        metadata_path = meeting_path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        chunk_count = metadata["chunk_count"]

        if chunk_count == 0:
            return MeetweenTaskResponse(
                type="catch_me_up",
                resultType="text",
                room_name=request.room_name,
                meeting_id=meeting_id,
                result="No audio recorded yet. The meeting just started.",
                language=request.language,
                participant_id=request.participant_id
            )

        logger.info(f"Processing catch-up for meeting {meeting_id}: {chunk_count} chunks, {metadata['total_duration']:.1f}s")
        if request.max_duration:
            logger.info(f"Max duration limit: {request.max_duration}s")

        # Combine all chunks into one audio file
        combined_audio = AudioSegment.empty()
        successful_chunks = 0

        for chunk_num in range(1, chunk_count + 1):
            chunk_path = meeting_path / f"chunk_{chunk_num:04d}.webm"
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

                    # Limit duration if specified
                    if request.max_duration and len(combined_audio) / 1000.0 > request.max_duration:
                        combined_audio = combined_audio[:int(request.max_duration * 1000)]
                        logger.info(f"Reached max duration limit, stopping at chunk {chunk_num}")
                        break

                except Exception as e:
                    # Skip corrupted chunks and continue
                    logger.warning(f"Skipping corrupted chunk {chunk_path.name}: {e}")
                    continue

        if len(combined_audio) == 0:
            return MeetweenTaskResponse(
                type="catch_me_up",
                resultType="text",
                room_name=request.room_name,
                meeting_id=meeting_id,
                result=f"No valid audio found. Recording contains {chunk_count} chunks but none could be decoded. The meeting may have just started.",
                language=request.language,
                participant_id=request.participant_id
            )

        logger.info(f"Successfully combined {successful_chunks}/{chunk_count} chunks")

        # Convert to mono, 16kHz
        combined_audio = combined_audio.set_channels(1).set_frame_rate(16000)

        # Get raw PCM data
        audio_array = np.array(combined_audio.get_array_of_samples(), dtype=np.int16)
        audio_pcm = audio_array.tobytes()

        duration_seconds = len(combined_audio) / 1000.0
        logger.info(f"Combined audio: {len(audio_pcm)} bytes, {duration_seconds:.1f}s at 16kHz")

        # Generate summary
        summary = summarize_audio(
            audio_bytes=audio_pcm,
            sample_rate=16000,
            engine=EngineManager.get_lmm(),
            language=request.language or "en-US"
        )

        return MeetweenTaskResponse(
            type="catch_me_up",
            resultType="text",
            room_name=request.room_name,
            meeting_id=meeting_id,
            result=summary,
            language=request.language,
            participant_id=request.participant_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Catch-up error for meeting {request.meeting_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate catch-up summary: {str(e)}"
        )

