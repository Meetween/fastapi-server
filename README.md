# Websocket Server

A FastAPI-based server that's compatible with both Jigasi Whisper and Meetween transcription protocols.

## Installation

1. **Install requirements**
   ```
   pip install -r requirements.txt
   ```

2. **Ensure Triton Inference Server is running**
   The FastAPI server connects to a Triton server for model inference.
   See the Triton repo for setup instructions.

3. **Run the Server**:
   ```bash
   python main.py
   ```
   alternatively to set the port
   ```bash
   uvicorn main:app --port $PORT (--reload)
   ```

## API Endpoints
For details on the loadout for requests and response, please refer to `models.py`

### Captions

###  Jigasi Live Captions (WebSockets)

Compatible with the custom `MeetweenTranscriptionService`.

**Endpoint**: `/ws/{meeting_id}`

**Request**: `MeetweenTranscriptionRequest`

**Response** `MeetweenTranscriptionResponse`


### Rest API

**Endpoint**: `/api/caption`

**Request**: Will change to pydantic model in the future
  ```json
  {
    "audio": File
    "language": str 
    "participant_id": Optional[str]
    "session_id": Optional[str]
  }
  ```

**Response** `MeetweenTranscriptionResponse`, 

### Transcription

**Endpoint**: `/api/transcribe`

**Request**: Will change to pydantic model in the future
  ```json
  {
    "audio": File
    "language": str 
    "participant_id": Optional[str]
  }
  ```

### Summary

**Endpoint**: `/api/summarize`

**Request**: `MeetweenTaskRequest`, `MeetweenAudioRequest`, `MeetweenTextRequest`

**Response** `MeetweenTaskResponse`

### Catchup

**Endpoint**: `/api/catchup`

**Request**: `MeetweenCatchupRequest`

**Response** `MeetweenTaskResponse`

### Agenda Tracking

**Endpoint**: `/api/agenda-tracking`

**Request**: `MeetweenAgendaTrackingRequest`

**Response**: `MeetweenAgendaTrackingResponse`

## Configuration

### Jigasi Configuration

```properties
# In jigasi configuration
JIGASI_CUSTOM_SERVICE=org.jitsi.jigasi.transcription.MeetweenTranscriptionService
org.jitsi.jigasi.transcription.meetween.websocket_url=ws://localhost:8000/ws
```