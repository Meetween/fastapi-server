# Meetween FastAPI Server

[![Python](https://img.shields.io/badge/Python-3.14%2B-green)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async%20API-009688)](https://fastapi.tiangolo.com/)
[![Triton](https://img.shields.io/badge/Triton-Inference%20Server-blue)](https://github.com/triton-inference-server/server)

Triton-backed meeting intelligence backend: real-time captions (WebSocket), HTTP transcription, speech and text summarization, catch-up summaries, meeting chat (Q&A), and agenda tracking.

Inference is always through **NVIDIA Triton** (`TritonClient` in `butler/triton_client.py`). You choose **which** model and URL to use via **`MODEL_PROFILE`** and `model_profiles.yml` (no automatic model discovery).

## Contents

- [Mission](#mission)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Environment and Configuration](#environment-and-configuration)
- [Model Profiles and Triton Protocols](#model-profiles-and-triton-protocols)
- [API Overview](#api-overview)
- [Recordings and Storage Paths](#recordings-and-storage-paths)
- [Jigasi Integration](#jigasi-integration)
- [Development](#development)
- [Operational Notes](#operational-notes)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Mission

- **Live captions**: WebSocket streaming aligned with Jigasi / Meetween-style clients.
- **Batch intelligence**: Transcribe and summarize audio or text over HTTP.
- **Catch-up**: Summarize the **in-memory meeting transcript** after duration checks against **stored recording metadata** (not a raw-audio re-ingest for that endpoint).
- **Chat**: Q&A using **server-side transcript** and **per-participant history** only.
- **Agenda**: Infer item statuses from the transcript; strict JSON contract.
- **Operations**: Configurable retention, scheduled cleanup, health endpoints.

## Key Features

- FastAPI + WebSocket (`api/routes/caption.py`).
- Single Triton client per process (`core/engine_manager.py`).
- **Model profiles** in `model_profiles.yml` merged into settings at startup (`config.py` → `model_post_init`).
- Prompts: phrase lists in `butler/prompts/INPUTS_TEXT_LIST.json` (ASR, `SSUM`, `TSUM`); chat/agenda templates in `butler/prompts/*.yml` via `utils/prompt_template.py`.
- Recording helpers: `main.py` Form/File endpoints and `utils/audio_recorder.py` (WebSocket-driven) — see [Recordings and Storage Paths](#recordings-and-storage-paths).
- Daily recording cleanup (APScheduler, `02:00` UTC).
- Pytest suite under `tests/` (see `tests/README.md`).

## How It Works

1. **Startup** (`main.py`): `EngineManager.initialize()` builds one `TritonClient` from `settings.triton` (after profile merge). If Triton is unreachable, startup fails.
2. **HTTP JSON APIs** validate bodies with Pydantic models in `models.py`.
3. **Audio** is decoded with **pydub** (FFmpeg) where needed, then **mono 16 kHz** PCM for model input (`api/routes/transcription.py`, `summary.py`, etc.).
4. **Inference**: `TritonClient._generate_text` POSTs to `{TRITON_URL}/v2/models/{model_name}/infer` with `TEXT_PROMPT`, optional `AUDIO_BYTES`, and scalar generation fields (`butler/triton_client.py`).
5. **Sessions**: `core/session_manager.py` holds transcript (from final caption segments), WebSocket handles, caption handlers, and per-participant chat history.
6. **Chat** (`api/routes/chat.py`): prompt is built from **`SessionManager` transcript only**.

## Project Structure

```text
fastapi-server/
├── main.py                 # App, lifespan, recording Form endpoints
├── config.py               # Settings + model_profiles.yml merge
├── models.py               # Pydantic request/response types
├── model_profiles.yml      # Named Triton presets
├── api/routes/
│   ├── agenda.py           # POST /api/agenda-tracking
│   ├── caption.py          # WebSocket + /api/caption
│   ├── chat.py             # POST /api/chat
│   ├── config.py           # GET /api/config/recording-retention-days
│   ├── health.py           # GET /, GET /health
│   ├── summary.py          # POST /api/summarize, /summarize-download, /catchup
│   └── transcription.py    # POST /api/transcribe
├── butler/
│   ├── jitsi_api_functions.py  # transcribe, caption, summarize_* → Triton
│   ├── triton_client.py
│   └── prompts/            # INPUTS_TEXT_LIST.json, *.yml
├── core/
│   ├── engine_manager.py
│   └── session_manager.py
├── utils/
└── tests/
```

## Requirements

- **Python** `>=3.14` (`pyproject.toml` → `requires-python`)
- **FFmpeg** (for pydub decoding)
- A running **Triton** server and a deployed model compatible with this client (see [Model Profiles and Triton Protocols](#model-profiles-and-triton-protocols))

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

Copy and edit `.env.example` if your runner loads dotenv, **or** export variables.

**Recommended:** set only the profile and let `model_profiles.yml` supply Triton URL, model name, and protocol:

```bash
export MODEL_PROFILE=qwen25_omni
```

Other valid profile names are in the table below (`speechlmm`, `voxtral_mini_3b_2507`, etc.).

Optional: override individual Triton fields with `TRITON_*` env vars **before** merge; the profile still replaces any keys it defines.

### 3) Run

```bash
python main.py
```

Uses `settings.host` (`0.0.0.0`) and `settings.port` (`8000`) from `config.py`. To use another port without changing code:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

### 4) Verify service health

```bash
curl -s http://localhost:8000/health
```

Interactive OpenAPI docs are also available when the server is running:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment and Configuration

Settings are defined in **`config.py`** (`Settings` + nested `TritonSettings`). Env vars on `TritonSettings` use the `TRITON_*` names below. **`host` and `port` are not read from the environment** in `Settings`; use `uvicorn --port` or change defaults in code.

### Application

| Name | Default | Source | Purpose |
| --- | --- | --- | --- |
| `MODEL_PROFILE` | `speechlmm` | env | Key in `model_profiles.yml` → `profiles` |
| `MODEL_PROFILES_PATH` | `model_profiles.yml` | env | YAML path; relative paths resolve next to `config.py` |
| `host` | `0.0.0.0` | code | `python main.py` bind host |
| `port` | `8000` | code | `python main.py` bind port |
| `log_level` | `INFO` | code | Logging |
| `recordings_base_path` | `/mnt/recordings` | code | Base path for stored chunks |
| `recording_format` | `wav` | code | Chunk file extension / format hint |
| `recording_retention_days` | `7` | code | `StorageManager.cleanup_old_recordings` |
| `min_catchup_duration_seconds` | `60` | code | Minimum meeting duration for `/api/catchup` |

### Triton (env-aware nested model)

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRITON_URL` | `http://localhost:9000` | Triton HTTP base |
| `TRITON_MODEL_NAME` | `qwen25_omni` | Model repository name |
| `TRITON_TIMEOUT` | `1800` | HTTP client timeout (seconds) |
| `TRITON_MAX_CONTEXT_CHARS` | `300000` | Transcript truncation budget (chat) |
| `TRITON_MAX_HISTORY_TURNS` | `400` | History turn cap (`SessionManager`) |
| `TRITON_MAX_HISTORY_CHARS` | `600000` | Chat history character cap |
| `TRITON_PROMPT_PROTOCOL` | `speechlmm_plain` | See [protocols](#model-profiles-and-triton-protocols) |
| `TRITON_CAPTION_CHARS` | `60` | Passed in JSON mode for `caption` task |

## Model Profiles and Triton Protocols

Profiles live in **`model_profiles.yml`**. At startup, `settings.triton` is replaced **field-by-field** by `profiles.<MODEL_PROFILE>.triton` when that block exists.

| Profile key | Typical `TRITON_URL` (example) | `model_name` | `prompt_protocol` |
| --- | --- | --- | --- |
| `speechlmm` | `http://localhost:9100` | `speechlmm` | `speechlmm_plain` |
| `qwen25_omni` | `http://localhost:9000` | `qwen25_omni` | `general_json` |
| `voxtral_mini_3b_2507` | `http://localhost:9000` | `voxtral_mini_3b_2507` | `general_json` |

**Protocols** (`butler/triton_client.py`):

- **`speechlmm_plain`**: `TEXT_PROMPT` is the raw instruction string. For **text-only** calls (no audio tensor), the client attaches a **short silent WAV** so Triton still receives `AUDIO_BYTES` (SpeechLMM-style backends).
- **`general_json`**: `TEXT_PROMPT` is a JSON string: `{"mode": "<task>", "prompt": "...", "format": "verbatim"}`; for `mode == "caption"`, `caption_chars` is included. Task/mode comes from the `task=` argument to `_generate_text` (e.g. `transcribe`, `caption`, `chat`, `SSUM`, `TSUM`).

**Multi-instance example** (different ports, same repo):

```bash
MODEL_PROFILE=voxtral_mini_3b_2507 uvicorn main:app --host 0.0.0.0 --port 8000
MODEL_PROFILE=qwen25_omni          uvicorn main:app --host 0.0.0.0 --port 8001
MODEL_PROFILE=speechlmm            uvicorn main:app --host 0.0.0.0 --port 8002
```

## API Overview

Authoritative request/response shapes: **`/docs`**, **`models.py`**.

### Task types (`models.taskTypes`)

| `type` value | Used for |
| --- | --- |
| `transcription` | Transcribe |
| `SSUM` | Speech summary (audio) |
| `TSUM` | Text summary |
| `catch_me_up` | Catch-up |
| `chat` | Chat |
| `agendaTracking` | Agenda (separate request model) |

### HTTP routes (prefixes from routers)

| Method | Path | Module | Notes |
| --- | --- | --- | --- |
| GET | `/` | `health.py` | Service + `model_profile`, Triton URL/name, `prompt_protocol` |
| GET | `/health` | `health.py` | Liveness-style JSON |
| POST | `/api/transcribe` | `transcription.py` | JSON body; audio is **base64** in `MeetweenAudioRequest`, not multipart |
| POST | `/api/summarize` | `summary.py` | Audio (`SSUM`), text (`TSUM`), or recording-based task request |
| POST | `/api/summarize-download` | `summary.py` | Same dispatcher, returns `text/plain` file |
| POST | `/api/catchup` | `summary.py` | `MeetweenCatchupRequest`; transcript + metadata under `room_name`/`meeting_id` |
| POST | `/api/chat` | `chat.py` | `MeetweenTaskRequest` with `type: chat`, `prompt` required |
| POST | `/api/agenda-tracking` | `agenda.py` | `MeetweenAgendaTrackingRequest`; `room_id` + transcript |
| GET | `/api/config/recording-retention-days` | `config.py` | Returns `recording_retention_days` |
| POST | `/api/recording/start` | `main.py` | `multipart/form-data` — see [paths](#recordings-and-storage-paths) |
| POST | `/api/recording/chunk` | `main.py` | Multipart chunk upload |
| POST | `/api/recording/stop` | `main.py` | Stop / optional delete |
| WS | `/ws/{meeting_id}` | `caption.py` | JSON text frames; see below |

### WebSocket captions

- **URL**: `ws://<host>:<port>/ws/{meeting_id}` — the **`meeting_id` is a path parameter** (not a single static `/ws` URL without it).
- Client sends **JSON text** messages. For audio chunks, `type` must be **`audio`** and the payload must validate as **`MeetweenTranscriptionRequest`** (`type`, `audio` base64, `meeting_id`, `room_name`, optional `language`, `participant_id`).
- Server responds with **`MeetweenTranscriptionResponse`** or **`MeetweenErrorResponse`** JSON.
- Final (non-interim) caption text is appended to **`SessionManager`** transcript for chat/catch-up.

### Generation limits (`max_new_tokens`)

- On `MeetweenTaskRequest`, **`max_new_tokens`** is constrained to **256–32768** (`models.py`).
- **`TSUM`** and **`get_summary_text_from_transcript`** pass `max_new_tokens` through to `summarize_text` → Triton.
- **`SSUM`** (audio) calls `summarize_audio` **without** passing `max_new_tokens` from the route today; the backend uses the **`max_tokens` default inside `summarize_audio`** (`2048` in `butler/jitsi_api_functions.py`). Sending `max_new_tokens` in JSON for SSUM does not change that until the route is wired.

### Catch-up (`POST /api/catchup`)

1. Requires **`recordings_base_path / room_name / meeting_id`** to exist with valid **`metadata.json`** (chunk count, `total_duration`).
2. Enforces **`min_catchup_duration_seconds`** against the effective duration window (optionally capped by `max_duration`).
3. Loads transcript from **`SessionManager.get_transcript`**; if empty → **404**.
4. Optionally trims transcript by character ratio when `max_duration` is set (`summary.py`).

## Recordings and Storage Paths

There are **two different on-disk layouts** in the current code:

| Flow | Path pattern | Used by |
| --- | --- | --- |
| **A** | `{recordings_base_path}/{meeting_id}/` | `main.py` → `/api/recording/start`, `chunk`, `stop` (Form `meeting_id` only) |
| **B** | `{recordings_base_path}/{room_name}/{meeting_id}/` | `AudioRecorder` / WebSocket captions; `transcription.py` and `summary.py` when loading chunks from disk; `StorageManager.cleanup_old_recordings` iterates room folders |

**Catch-up, summarize-from-recording, and transcribe-from-recording** expect **layout B**. If you only use **layout A** (`main.py` recording API), those HTTP flows will not find metadata under `room_name/meeting_id`.

## Jigasi Integration

Example properties (adjust host, port, and how your Jigasi build substitutes **meeting id** in the WebSocket path):

```properties
JIGASI_CUSTOM_SERVICE=org.jitsi.jigasi.transcription.MeetweenTranscriptionService
org.jitsi.jigasi.transcription.meetween.websocket_url=ws://localhost:8000/ws
```

Your client implementation must resolve a URL of the form **`.../ws/<meeting_id>`** so it matches FastAPI’s **`/ws/{meeting_id}`** route.

## Development

### Tests

```bash
pytest
```

More examples: **`tests/README.md`**.

```bash
pytest -m "not slow"
pytest tests/api/test_summary.py
pytest -k "chat"
```

Coverage is configured in `pyproject.toml` for `api`, `butler`, `core`.

### Local development tips

- Use `uvicorn main:app --reload` while editing routes.
- Keep Triton and the model named in your profile reachable before starting the app.

## Operational Notes

- **Cleanup**: APScheduler runs `StorageManager.cleanup_old_recordings` daily at **02:00 UTC**.
- **Shutdown**: `shutdown_event` closes scheduler, `SessionManager.cleanup`, and `EngineManager.cleanup` (HTTP client).

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Startup fails after “Connecting to Triton” | Wrong `TRITON_URL` / model name, or Triton not ready |
| `503` “model not loaded” | `EngineManager` not initialized |
| Chat always empty context | No caption WebSocket session updating transcript; **`context` on the request is ignored** |
| Catch-up 404 “no transcript” | Transcript never populated for that `meeting_id` + `room_name` |
| Catch-up 404 “meeting path” | Missing **`recordings_base_path/room_name/meeting_id`** (layout B) |
| SSUM slow / long output | Default `max_tokens` in `summarize_audio`; not tied to request `max_new_tokens` today |

## Contributing

- Keep PRs focused; add or update tests for behavior changes.
- Run **`pytest`** before opening a PR.
- Update **this README** and **`.env.example`** when configuration or routes change.
- Prefer backward-compatible API changes unless the team agrees on a break.

## License

No license file is currently included. Add a `LICENSE` file before publishing this repository as open source.
