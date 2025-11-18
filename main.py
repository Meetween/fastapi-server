"""
Simple Whisper Server compatible with both Whisper and Meetween protocols
Supports both WebSocket streaming and single transcription requests
"""
import json
import logging
import uuid
from typing import Dict, Set
import random
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from models import (
    WhisperAudioMessage,
    WhisperResponse,
    MeetweenAudioRequest,
    MeetweenTranscriptionResponse,
    MeetweenErrorResponse,
    TranscriptionResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meetween FastAPI Server", version="1.0.0")
# Global whisper model (can be loaded on startup)
whisper_model = None

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}


def load_whisper_model(model_name: str = "base"):
    """Load Whisper model on startup"""
    global whisper_model
    logger.info(f"Loading Whisper model: {model_name}")
    # import whisper
    # whisper_model = whisper.load_model(model_name)
    logger.info("Whisper model loaded successfully")


async def transcribe_mock_audio(audio_data: bytes, language: str = "en") -> TranscriptionResult:
    """
    Transcribe audio data using Whisper model
    Currently returns mock data for testing
    """
    try:
        logger.info(f"Transcribing {len(audio_data)} bytes, language: {language}")

        # Mock transcription for testing
        import random
        texts = [
            "Hello, this is a test transcription.",
            "The audio has been transcribed succesfully.",
            "Lorem ipsum dolor sit amet.",
            "This is dummy text."
        ]

        return TranscriptionResult(
            text=random.choice(texts),
            confidence=0.95,
            language=language
        )

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return TranscriptionResult(
            text="Error in transcription",
            confidence=0.0,
            language=language,
            error=str(e)
        )


# @app.on_event("startup")
# async def startup_event():
#     """Initialize server"""
#     logger.info("Server started in mock mode")
#     # Uncomment to load actual Whisper model:
#     # load_whisper_model("base")


# ===== WHISPER PROTOCOL WEBSOCKET =====

@app.websocket("/ws/{connection_id}")
async def whisper_websocket(websocket: WebSocket, connection_id: str):
    """
    Whisper WebSocket endpoint
    Expects binary data with 60-byte header: "participant_id|language" + audio data
    """
    await websocket.accept()
    active_connections[connection_id] = websocket

    logger.info(f"Whisper WebSocket connected: {connection_id}")

    try:
        while True:
            # Receive binary data
            data = await websocket.receive_bytes()

            if len(data) <= 60:
                logger.debug(f"Received small packet: {len(data)} bytes")
                continue

            try:
                # Parse using Pydantic model
                message = WhisperAudioMessage.from_binary(data)
                logger.info(f"Processing audio from '{message.participant_id}' in '{message.language}'")


                # Transcribe audio
                result = await transcribe_mock_audio(message.audio_data, message.language)

                if not result.has_error and result.text:
                    # # Send interim result
                    # interim = WhisperResponse(
                    #     type="interim",
                    #     participant_id=message.participant_id,
                    #     text=result.text,
                    #     variance=0.1
                    # )
                    # await websocket.send_text(interim.model_dump_json())

                    # Send final result
                    final = WhisperResponse(
                        type="final",
                        participant_id=message.participant_id,
                        text=result.text,
                        variance=0.0
                    )
                    await websocket.send_text(final.model_dump_json())

            except ValueError as e:
                logger.warning(f"Invalid Whisper message format: {e}")
                continue

    except WebSocketDisconnect as e:
        logger.info(f"Whisper WebSocket disconnected: {connection_id}, code: {e.code}")
    except Exception as e:
        logger.error(f"Whisper WebSocket error: {e}", exc_info=True)
    finally:
        active_connections.pop(connection_id, None)
        logger.info(f"Cleaned up connection {connection_id}")


# ===== MEETWEEN PROTOCOL WEBSOCKET =====

@app.websocket("/transcribe_mock/{connection_id}")
async def meetween_websocket(websocket: WebSocket, connection_id: str):
    """
    Meetween WebSocket endpoint
    Messageformat:
    {
      "type": "audio",
      "participant_id": "participant123",
      "language": "en",
      "audio": "<base64-encoded-audio>"
    }
    """
    await websocket.accept()
    active_connections[f"meetween_{connection_id}"] = websocket

    logger.info(f"Meetween WebSocket connected: {connection_id}")
    mock_uuid = uuid.uuid4()
    try:
        while True:
            # Receive JSON message
            message = await websocket.receive_text()

            try:
                # Parse JSON
                data = json.loads(message)
                msg_type = data.get("type")

                # Parse based on type
                if msg_type == "audio":
                    # Streaming audio request
                    try:
                        request = MeetweenAudioRequest(**data)
                        logger.info(f"Processing audio for participant '{request.participant_id}'")

                        # Get audio bytes
                        audio_data = request.get_audio_bytes()

                        # Transcribe
                        result = await transcribe_mock_audio(audio_data, request.language)

                        if not result.has_error and result.text:
                            if (random.randint(0,4) % 3) != 0:
                                # Send interim result
                                logger.info("Sending interim result")
                                interim = MeetweenTranscriptionResponse(
                                    participant_id=request.participant_id,
                                    transcript=result.text,
                                    is_final=False,
                                    confidence=result.confidence,
                                    transcript_id=mock_uuid
                                )
                                await websocket.send_text(interim.model_dump_json())
                            else:
                                logger.info("Sending final result")
                                # Send final result
                                final = MeetweenTranscriptionResponse(
                                    participant_id=request.participant_id,
                                    transcript=result.text,
                                    is_final=True,
                                    confidence=result.confidence,
                                    transcript_id=mock_uuid
                                )
                                mock_uuid = uuid.uuid4()
                                await websocket.send_text(final.model_dump_json())

                    except ValidationError as e:
                        error = MeetweenErrorResponse(
                            message=f"Invalid audio request: {e}",
                            code="INVALID_REQUEST"
                        )
                        print(e)
                        await websocket.send_text(error.model_dump_json())

                # elif msg_type == "single_transcription":
                #     # Single transcription request
                #     try:
                #         request = MeetweenSingleRequest(**data)
                #         logger.info(f"Processing single transcription: {request.request_id}")

                #         # Get audio bytes
                #         audio_data = request.get_audio_bytes()

                #         # Transcribe
                #         result = await transcribe_mock_audio(audio_data, request.language)

                #         # Send response
                #         response = MeetweenSingleResponse(
                #             request_id=request.request_id,
                #             transcript=result.text,
                #             confidence=result.confidence
                #         )
                #         await websocket.send_text(response.model_dump_json())

                #     except ValidationError as e:
                #         error = MeetweenErrorResponse(
                #             message=f"Invalid single transcription request: {e}",
                #             code="INVALID_REQUEST"
                #         )
                #         await websocket.send_text(error.model_dump_json())

                else:
                    logger.warning(f"Unknown message type: {msg_type}")
                    error = MeetweenErrorResponse(
                        message=f"Unknown message type: {msg_type}",
                        code="UNKNOWN_TYPE"
                    )
                    await websocket.send_text(error.model_dump_json())

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON: {e}")
                error = MeetweenErrorResponse(
                    message=f"Invalid JSON: {e}",
                    code="INVALID_JSON"
                )
                await websocket.send_text(error.model_dump_json())

    except WebSocketDisconnect:
        logger.info(f"Meetween WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Meetween WebSocket error: {e}", exc_info=True)
    finally:
        active_connections.pop(f"meetween_{connection_id}", None)
        logger.info(f"Cleaned up Meetween connection {connection_id}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
