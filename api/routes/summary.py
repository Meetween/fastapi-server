import json
import logging
import tempfile
import os

import numpy as np
from pydub import AudioSegment
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse

from core import EngineManager
from utils import StorageManager, pydub_to_pcm16
from config import settings
from models import (
    MeetweenTaskResponse,
    MeetweenAudioRequest,
    MeetweenTextRequest,
    MeetweenTaskRequest,
    MeetweenCatchupRequest,
    LMMRequest,
    TASK_TYPE_TEXT_SUMMARY,
    TASK_TYPE_SPEECH_SUMMARY,
)
from butler import summarize_audio, summarize_text
from core import SessionManager

router = APIRouter(prefix="/api", tags=["summary"])

logger = logging.getLogger(__name__)


def _load_meeting_metadata(meeting_path):
    metadata_path = meeting_path / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No audio found to summarize."
        )
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error("Invalid metadata.json at %s", metadata_path)
        raise HTTPException(
            status_code=400,
            detail="Meeting metadata is invalid."
        )


def _validate_summary_task_type(
    request: LMMRequest,
    expected_type: str | None = None,
) -> str:
    if request.type not in [TASK_TYPE_TEXT_SUMMARY, TASK_TYPE_SPEECH_SUMMARY]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid summary task type '{request.type}' for this payload. "
                f"Expected '{TASK_TYPE_TEXT_SUMMARY}' or '{TASK_TYPE_SPEECH_SUMMARY}'."
            ),
        )
    if expected_type is not None and request.type != expected_type:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid summary task type '{request.type}' for this endpoint. "
                f"Expected '{expected_type}'."
            ),
        )
    return request.type


async def get_summary_text(request: LMMRequest) -> str:
    """
    Core summarization dispatcher for summarization requests.
    Takes a LMMRequest and returns the summary text string.
    
    Args:
        request: LMMRequest
        
    Returns:
        Summary text
    """
    engine = EngineManager.get_lmm()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="SpeechLMM model not loaded"
        )

    # Handle audio input payload.
    if isinstance(request, MeetweenAudioRequest):
        _validate_summary_task_type(request, expected_type=TASK_TYPE_SPEECH_SUMMARY)
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
            audio_pcm = pydub_to_pcm16(audio_segment)
            logger.info(f"Decoded audio: {len(audio_pcm)} bytes at 16000Hz")

            # Generate summary
            return await summarize_audio(
                audio_bytes=audio_pcm,
                sample_rate=16000,
                engine=engine,
                language=request.language or "en-US",
                prompt=request.prompt,
                context=request.context,
                history=request.history
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    # Handle text input
    if isinstance(request, MeetweenTextRequest):
        _validate_summary_task_type(request, expected_type=TASK_TYPE_TEXT_SUMMARY)
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text data")
        return await summarize_text(
            engine=engine,
            text=request.text,
            language=request.language or "en-US",
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
        )


    if isinstance(request, MeetweenTaskRequest):
        if request.type == TASK_TYPE_TEXT_SUMMARY:
            _validate_summary_task_type(request, expected_type=TASK_TYPE_TEXT_SUMMARY)
            return await get_summary_text_from_transcript(request)
        _validate_summary_task_type(request, expected_type=TASK_TYPE_SPEECH_SUMMARY)

        meeting_id = request.meeting_id
        room_name = request.room_name
        meeting_path = settings.recordings_base_path / room_name / meeting_id

        if not meeting_path.exists():
            logger.error(f"No audio found. Can't summarize. Meeting: {room_name}/{meeting_id}")
            raise HTTPException(
                status_code=404,
                detail="No audio found to summarize."
            )

        metadata = _load_meeting_metadata(meeting_path)

        chunk_count = metadata["chunk_count"]

        if chunk_count == 0:
            logger.error(f"No valid audio found. Can't summarize. Meeting: {room_name}/{meeting_id}")
            raise HTTPException(
                status_code=400,
                detail="No valid audio found to summarize."
            )

        logger.info(f"Summarizing meeting {meeting_id}: {chunk_count} chunks, {metadata['total_duration']:.1f}s")

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
            logger.error(f"No valid audio found. Recording contains {chunk_count} chunks but none could be decoded.")
            raise HTTPException(
                status_code=400,
                detail="No valid audio found. Audio could not be decoded"
            )

        logger.info(f"Successfully combined {successful_chunks}/{chunk_count} chunks")

        # Convert to mono, 16kHz
        combined_audio = combined_audio.set_channels(1).set_frame_rate(16000)

        # Get raw PCM data
        audio_array = np.array(combined_audio.get_array_of_samples(), dtype=np.int16)
        audio_pcm = audio_array.tobytes()

        duration_seconds = len(combined_audio) / 1000.0
        logger.info(f"Combined audio: {len(audio_pcm)} bytes, {duration_seconds:.1f}s at 16kHz")

        return await summarize_audio(
            audio_bytes=audio_pcm,
            sample_rate=16000,
            engine=engine,
            language=request.language or "en-US",
            prompt=request.prompt,
            context=request.context,
            history=request.history,
        )


async def get_summary_text_from_transcript(
    request: LMMRequest,
    transcript_text: str | None = None,
) -> str:
    """
    Generate text summary from server-side session transcript.
    
    Args:
        request: LMMRequest
        
    Returns:
        Summary text
    """
    engine = EngineManager.get_lmm()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Engine not loaded"
        )

    transcript = transcript_text or SessionManager.get_transcript(request.meeting_id, request.room_name) or ""
    if not transcript.strip():
        raise HTTPException(
            status_code=404,
            detail="No transcript found to summarize."
        )

    return await summarize_text(
        engine=engine,
        text=transcript,
        language=request.language or "en-US",
        max_new_tokens=request.max_new_tokens,
        do_sample=request.do_sample,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
    )


@router.post("/summarize", response_model=MeetweenTaskResponse)
async def api_summarize(request: LMMRequest = Body(...)):
    """
    Supports three input types:
    - audio: Base64-encoded audio data (MeetweenAudioRequest)
    - text: Plain text input (MeetweenTextRequest)
    - Neither: Basic request (MeetweenTaskRequest), use local recording

    Returns a summary with task metadata
    """
    try:
        summary = await get_summary_text(request)
        
        return MeetweenTaskResponse(
            type=request.type,
            resultType="text",
            room_name=request.room_name,
            meeting_id=request.meeting_id,
            result=summary,
            language=request.language,
            participant_id=request.participant_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@router.post("/summarize-download")
async def api_summarize_download(request: LMMRequest = Body(...)):
    """
    Download the summarized text as a .txt file.
    
    Supports three input types:
    - audio: Base64-encoded audio data (MeetweenAudioRequest)
    - text: Plain text input (MeetweenTextRequest)
    - Neither: Basic request (MeetweenTaskRequest), use local recording

    Returns a downloadable .txt file containing the summary.
    """
    try:
        summary = await get_summary_text(request)
        summary_bytes = summary.encode('utf-8')
        filename = f"summary_{request.room_name}_{request.meeting_id}.txt"
        return StreamingResponse(
            iter([summary_bytes]),
            media_type='text/plain',
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization download error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization download failed: {str(e)}")


@router.post("/catchup", response_model=MeetweenTaskResponse)
async def api_catchup(request: MeetweenCatchupRequest = Body(...)):
    """
    Generate a "catch me up" summary from the server-side meeting transcript.

    This endpoint uses recording metadata for duration gating, and summarizes the
    in-memory transcript for the meeting.

    - **meeting_id**: Meeting identifier to load recordings from
    - **max_duration**: Optional maximum meeting duration to summarize (best effort)

    Returns a catch_me_up type response with summary
    """
    if not EngineManager.get_lmm():
        raise HTTPException(
            status_code=503,
            detail="SpeechLMM model not loaded"
        )

    try:
        meeting_id = request.meeting_id
        room_name = request.room_name
        meeting_path = settings.recordings_base_path / room_name / meeting_id

        if not meeting_path.exists():
            logger.error(f"No audio found for catching up. Meeting: {room_name}/{meeting_id}")
            raise HTTPException(
                status_code=404,
                detail=f"The meeting just started. We have not received enough valid audio yet to generate a summary."
            )

        metadata = _load_meeting_metadata(meeting_path)

        chunk_count = metadata["chunk_count"]

        total_duration = float(metadata.get("total_duration", 0.0))
        if chunk_count == 0 or total_duration <= 0:
            logger.error(f"No audio received yet. The meeting just started. Meeting: {room_name}/{meeting_id}")
            raise HTTPException(
                status_code=404,
                detail=f"The meeting just started. We have not received enough valid audio yet to generate a summary."
            )

        logger.info(f"Processing catch-up for meeting {meeting_id}: {chunk_count} chunks, {total_duration:.1f}s")
        if request.max_duration:
            logger.info(f"Max duration limit: {request.max_duration}s")
        duration_seconds = total_duration
        if request.max_duration:
            duration_seconds = min(duration_seconds, request.max_duration)
        logger.info(f"Effective catch-up duration window: {duration_seconds:.1f}s")

        min_duration = settings.min_catchup_duration_seconds
        if duration_seconds < min_duration:
            logger.error(f"Not enough meeting duration ({duration_seconds:.0f}s). Can't generate catch-up summary. Requires :{min_duration // 60}min")
            raise HTTPException(
                status_code=400,
                detail="The meeting just started. Please wait a few more minutes for a catch-up summary.",
            )

        transcript = SessionManager.get_transcript(meeting_id=meeting_id, room_name=room_name) or ""
        if not transcript.strip():
            logger.error(f"No transcript found for catching up. Meeting: {room_name}/{meeting_id}")
            raise HTTPException(
                status_code=404,
                detail="No transcript found to summarize."
            )

        transcript_for_summary = transcript
        if request.max_duration and total_duration > 0:
            ratio = min(1.0, request.max_duration / total_duration)
            if ratio < 1.0:
                target_chars = max(1, int(len(transcript) * ratio))
                transcript_for_summary = transcript[-target_chars:]
                logger.info(
                    "Applying max_duration transcript window: ratio=%.3f chars=%d/%d",
                    ratio,
                    len(transcript_for_summary),
                    len(transcript),
                )

        summary = await get_summary_text_from_transcript(
            request=request,
            transcript_text=transcript_for_summary,
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

