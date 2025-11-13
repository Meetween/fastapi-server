# Websocket Server

A FastAPI-based server that's compatible with both Jigasi Whisper and Meetween transcription protocols.

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Server**:
   ```bash
   python main.py
   ```

   Or with uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints

### WebSocket Endpoints

#### 1. Whisper Protocol: `/ws/{connection_id}`

Compatible with Jigasi's `WhisperTranscriptionService`.

**Protocol**: Binary data with 60-byte header
- **Header Format**: `"participant_id|language"` (padded to 60 bytes)
- **Data**: Raw audio bytes following the header


#### 2. Meetween Protocol: `/transcribe_mock/{connection_id}`

Compatible with the custom `MeetweenTranscriptionService`.

**Protocol**: JSON metadata followed by binary audio

**Streaming Audio**:
```json
{"type": "audio", "participant_id": "user123", "language": "en", "audio": "<base64-encoded-audio>" }

```

**Single Transcription**:

Not implemented

## Response Formats

### Whisper Protocol Responses

**Interim Result**:
```json
{
  "type": "interim",
  "participant_id": "user123",
  "text": "Hello world",
  "variance": 0.1
}
```

**Final Result**:
```json
{
  "type": "final",
  "participant_id": "user123",
  "text": "Hello world",
  "variance": 0.0
}
```

### Meetween Protocol Responses

**Streaming Result**:
```json
{
  "type": "transcription",
  "participant_id": "user123",
  "transcript": "Hello world",
  "is_final": true,
  "confidence": 0.95
}
```

## Configuration

### Jigasi Configuration

**For Whisper Protocol**:
```properties
# In jigasi configuration
JIGASI_CUSTOM_SERVICE=org.jitsi.jigasi.transcription.WhisperTranscriptionService
org.jitsi.jigasi.transcription.whisper.websocket_url=ws://localhost:8000/ws
```

**For Meetween Protocol**:
```properties
# In jigasi configuration
JIGASI_CUSTOM_SERVICE=org.jitsi.jigasi.transcription.MeetweenTranscriptionService
org.jitsi.jigasi.transcription.meetween.websocket_url=ws://localhost:8000/transcribe
```
