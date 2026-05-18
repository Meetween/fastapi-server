# Meetween FastAPI Server

[![Python](https://img.shields.io/badge/Python-3.14%2B-green)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async%20API-009688)](https://fastapi.tiangolo.com/)
[![Triton](https://img.shields.io/badge/Triton-Inference%20Server-blue)](https://github.com/triton-inference-server/server)

Triton-backed meeting intelligence backend for real-time captions, transcription, summarization, catch-up, chat Q&A, and agenda tracking.

This service is designed for meeting products that need low-latency streaming plus post-meeting and in-meeting AI workflows, with FastAPI endpoints and model inference served through Triton (for example SpeechLMM, Qwen Omni 2.5, Voxtral, and compatible models).

## Contents

- [Mission](#mission)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Environment and Configuration](#environment-and-configuration)
- [API Documentation](#api-documentation)
- [Jigasi Integration](#jigasi-integration)
- [Development](#development)
- [Operational Notes](#operational-notes)
- [Contributing](#contributing)
- [License](#license)

## Mission

Provide a production-ready API layer that turns raw meeting audio and transcript context into useful meeting intelligence:

- Real-time captions over WebSocket for live sessions
- Primary live integration with Jigasi (`jitsi-docker-dev` frontend path)
- High-quality transcription and summarization via HTTP APIs
- "Catch me up" summaries from recorded chunks
- Context-aware meeting chat and agenda progress tracking
- Model profile and runtime configuration through environment variables

## Key Features

- FastAPI app with async endpoints and WebSocket support
- Triton inference integration with configurable model profiles and model backends
- Recording lifecycle APIs (`start`, `chunk`, `stop`) for meeting audio storage
- Prompt-template based chat and agenda logic
- Daily recording cleanup job via APScheduler
- Pytest suite for API and utility behavior

## How It Works

At runtime, the server receives live or uploaded meeting audio, normalizes it for model inference, sends requests to the configured Triton-hosted model, and returns structured API responses to clients.

1. Jigasi (used by the `jitsi-docker-dev` frontend flow) sends live audio chunks over `WS /ws/{meeting_id}`, while other clients can call task APIs over HTTP.
2. The server validates payloads with Pydantic models from `models.py`.
3. Audio is decoded and converted to mono 16 kHz PCM before inference.
4. `EngineManager` routes generation work to the configured Triton model profile.
5. `SessionManager` keeps meeting transcript and participant chat context in memory.
6. API routes return typed responses for captions, transcription, summaries, chat, and agenda tracking.

## Project Structure

```text
fastapi-server/
├── main.py
├── config.py
├── models.py
├── model_profiles.yml
├── api/
│   └── routes/
│       ├── agenda.py
│       ├── caption.py
│       ├── chat.py
│       ├── config.py
│       ├── health.py
│       ├── summary.py
│       └── transcription.py
├── butler/
├── core/
├── utils/
└── tests/
```

## Requirements

- Python 3.14+ (as declared in `pyproject.toml`)
- FFmpeg (required by `pydub` for audio decoding)
- Access to a running Triton inference server
- A compatible model served through Triton (for example SpeechLMM, Qwen Omni 2.5, Voxtral)

## Quick Start

### 1) Clone and install dependencies

```bash
git clone <your-repo-url>
cd fastapi-server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If FFmpeg is not already installed:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

### 2) Configure environment

Set environment variables as needed (defaults are defined in `config.py`):

```bash
export MODEL_PROFILE="speechlmm"
export MODEL_PROFILES_PATH="model_profiles.yml"
export TRITON_TIMEOUT="1800"
```

### 3) Start the server

```bash
python main.py
```

Or run with Uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4) Verify service health

```bash
curl http://localhost:8000/health
```

Interactive OpenAPI docs are also available when the server is running:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment and Configuration

Configuration is loaded through `pydantic-settings` in `config.py`.

### Core App Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `host` | `0.0.0.0` | Bind address used by `python main.py` |
| `port` | `8000` | Port used by `python main.py` |
| `log_level` | `INFO` | Uvicorn/application log level |
| `recordings_base_path` | `/mnt/recordings` | Base directory for meeting audio chunks |
| `recording_format` | `wav` | Stored chunk format |
| `recording_retention_days` | `7` | Retention window used by cleanup |
| `min_catchup_duration_seconds` | `60` | Minimum audio duration before catch-up summaries |

### Triton Settings

`url`, `model_name`, and `prompt_protocol` come from the selected entry in `model_profiles.yml` (via `MODEL_PROFILE`), not from env vars.

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRITON_TIMEOUT` | `1800` | Inference request timeout in seconds |
| `TRITON_MAX_CONTEXT_CHARS` | `300000` | Maximum transcript/context size |
| `TRITON_MAX_HISTORY_TURNS` | `400` | Maximum chat history turns |
| `TRITON_MAX_HISTORY_CHARS` | `600000` | Maximum chat history characters |
| `TRITON_CAPTION_CHARS` | `60` | Caption formatting length (profile YAML may override) |

### Model Profiles

Use `MODEL_PROFILE` + `MODEL_PROFILES_PATH` to switch Triton endpoint, model name, and prompt protocol without changing code.

Example profiles are defined in `model_profiles.yml`:

- `speechlmm`
- `qwen25_omni`

Profile values override only the Triton fields that are present in the selected profile.

## API Documentation

Use FastAPI's generated docs as the source of truth for endpoints, request bodies, and response schemas:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

Schema models are defined in `models.py`.

The API surface is organized by capability:

- Health and service status
- Live captioning over WebSocket
- Transcription and summarization tasks
- Catch-up summaries from meeting recordings
- Meeting chat and agenda tracking
- Recording lifecycle and runtime config

## Jigasi Integration

For custom Jigasi transcription service setup:

```properties
JIGASI_CUSTOM_SERVICE=org.jitsi.jigasi.transcription.MeetweenTranscriptionService
org.jitsi.jigasi.transcription.meetween.websocket_url=ws://localhost:8000/ws
```

Jigasi is the primary real-time integration path for transcript streaming in this project.

## Development

### Run tests

```bash
pytest
```

Useful options:

```bash
pytest -m "not slow"
pytest tests/api/test_summary.py
pytest -k "chat"
```

Coverage reports are configured in `pyproject.toml` and include:

- `api`
- `butler`
- `core`

### Local development tips

- Use `uvicorn ... --reload` during endpoint development
- Keep Triton running locally (or point to a remote environment)
- Use `tests/README.md` for test-suite conventions and examples

## Operational Notes

- The service schedules a daily recording cleanup job at `02:00 UTC`
- On shutdown, active session state and model resources are cleaned up
- Recording retention and minimum catch-up duration are configurable

## Contributing

Contributions are welcome. Please keep PRs focused, include tests for behavior changes, and preserve API/schema compatibility unless a breaking change is explicitly discussed.

Before opening a PR:

- Run `pytest`
- Update docs when behavior or configuration changes
- Avoid committing generated files such as coverage output
- Keep endpoint contracts backward compatible unless the change is intentional

## License

No license file is currently included. Add a `LICENSE` file before publishing this repository as open source.
