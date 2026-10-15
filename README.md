# Meetween FastAPI Server

[![Python](https://img.shields.io/badge/Python-3.12-green)](https://www.python.org/)

The meeting intelligence backend of Meetween. Jigasi streams meeting audio to it over a
websocket and gets live captions and translations back; clients call it over HTTP for
transcription, translation, summaries, catch up, chat over the transcript and agenda tracking.
Inference runs on a Triton server through a selectable model profile. MongoDB holds users and
meeting context; transcripts and sessions live in process memory.

## How it works

1. Jigasi opens `WS /ws/{meeting_id}` and sends PCM audio chunks. Other clients call `/api/*`
   with base64 audio or text, with a bearer token when authentication is on.
2. The request is validated by the Pydantic models in `models.py`.
3. Audio is decoded with pydub and ffmpeg and converted to mono 16 kHz PCM.
4. `EngineManager` sends the task to the Triton model selected by `MODEL_PROFILE`, over HTTP.
5. `SessionManager` keeps the transcript and the per participant chat history of each meeting.
6. The route returns a typed response, or pushes captions back on the websocket.

Recording chunks are stored under `RECORDINGS_BASE_PATH` and deleted daily at 02:00 UTC after
`RECORDING_RETENTION_DAYS`.

## Requirements

- Python 3.12, selected by `.python-version`, and [uv](https://docs.astral.sh/uv/)
- ffmpeg on the PATH, used by pydub
- a reachable Triton server with one of the models in `model_profiles.yml`
- MongoDB, `docker-compose.yml` starts one

## Run

```sh
uv sync
docker compose up -d mongo
RECORDINGS_BASE_PATH=/tmp/recordings uv run main.py
curl localhost:8000/health
```

The recordings path defaults to `/mnt/recordings`, the directory on the shared VM; the service
creates it at startup, so point it somewhere writable on a laptop. Interactive API docs at `http://localhost:8000/docs`. With `ENABLE_AUTH=true`, the default,
create the first user and log in:

```sh
uv run cli.py create-admin admin@example.com --password <password>
curl -X POST localhost:8000/api/auth/login -H 'content-type: application/json' \
  -d '{"email":"admin@example.com","password":"<password>"}'
```

## Configuration

Every field of `Settings` in `config.py` is an environment variable, read from the environment
or from a `.env` file next to `config.py`. The ones that matter to run the service:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PROFILE` | `speechlmm` | profile in `model_profiles.yml`: Triton URL, model name, prompt protocol |
| `MODEL_PROFILES_PATH` | `model_profiles.yml` | where the profiles are read from |
| `PORT` | `8000` | port used by `uv run main.py` |
| `ENABLE_AUTH` | `true` | bearer authentication on `/api/*` |
| `JWT_SECRET_KEY` | `change-me-in-production` | HS256 secret for access, refresh and Jitsi tokens |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | access token lifetime; an expired token gets `401` with `Access token expired` |
| `JITSI_MEET_BASE_URL` | unset | when set, `/api/auth/jitsi/issue` returns a ready join URL |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection |
| `RECORDINGS_BASE_PATH` | `/mnt/recordings` | where recording chunks are written |
| `RECORDING_RETENTION_DAYS` | `7` | retention of recording chunks |
| `TRITON_TIMEOUT` | `1800` | inference timeout in seconds |
| `LOG_LEVEL` | `INFO` | application and uvicorn log level |

Profiles shipped in `model_profiles.yml`: `speechlmm` (plain text prompt plus WAV),
`qwen25_omni` and `voxtral_mini_3b_2507` (JSON envelope). A profile overrides only the Triton
fields it lists.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| WS | `/ws/{meeting_id}` | audio in, captions and translations out, used by Jigasi |
| POST | `/api/transcribe` | full transcription of an audio payload |
| POST | `/api/translate` | text or speech translation |
| POST | `/api/summarize`, `/api/summarize-download` | meeting summary, as JSON or as a file |
| POST | `/api/catchup` | catch up summary from the stored recording |
| POST | `/api/chat` | question answering over the transcript and the chat history |
| POST | `/api/agenda-tracking` | status of each agenda item from the transcript |
| GET | `/api/config/recording-retention-days` | runtime configuration |
| POST | `/api/auth/login`, `/api/auth/refresh`, `/api/auth/jitsi/validate` | public auth endpoints |
| POST | `/api/auth/users`, `/api/auth/jitsi/issue` | user creation and Jitsi room token, authenticated |
| GET | `/api/auth/context/{room_name}` | meeting context, authenticated |
| GET | `/`, `/health` | liveness; `/health` reports the running version |

Request and response schemas are in `models.py` and in the generated docs.

## Jigasi integration

```properties
JIGASI_CUSTOM_SERVICE=org.jitsi.jigasi.transcription.MeetweenTranscriptionService
org.jitsi.jigasi.transcription.meetween.websocket_url=ws://<host>:8000/ws
```

Jigasi appends `/{meeting_id}` to `websocket_url`, so the value ends in `/ws`.

## Work on it

```sh
uv run pytest
uv run pytest tests/api/test_summary.py
uv run pytest -k chat
```

Tests mirror the routes, `tests/api/test_<route>.py`, with unit tests under `tests/unit/`.
Triton and MongoDB are mocked in `tests/conftest.py`, so the suite needs no GPU and no
database, only ffmpeg. The same command runs in CI on every pull request.

Branches, pull request titles and releases are in [CONTRIBUTING.md](CONTRIBUTING.md).
Deployment to the shared VM is in [deploy/README.md](deploy/README.md). Released versions are
in [CHANGELOG.md](CHANGELOG.md).

## License

Apache-2.0, see [LICENSE](LICENSE).
