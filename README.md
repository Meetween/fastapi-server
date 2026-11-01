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
- MongoDB, `docker-compose.yml` starts one for local development only. The deployment does not
  use this file; see `deploy/README.md`.

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
| `ASR_SILENCE_RMS` | `150` | caption websocket: chunks whose 20 ms frames all stay below this RMS (raw int16 units, 0 to 32767, 150 is about -47 dBFS) are not sent to the model; `0` turns the gate off |
| `ASR_SILENCE_MIN_ACTIVE_MS` | `100` | caption websocket: time at or above `ASR_SILENCE_RMS` a chunk needs to count as speech |
| `ASR_UTTERANCE_MODE` | `false` | caption websocket: join a participant's speech chunks until a pause, transcribe them once and store that as the transcript line; live captions keep their 2 s pace. Off stores one line per caption chunk |
| `ASR_UTTERANCE_MAX_S` | `10` | caption websocket: an utterance is closed and transcribed at this length; it also caps the buffer at about 32 KB per second per speaker |
| `ASR_UTTERANCE_IDLE_S` | `3` | caption websocket: an utterance is also closed when its speaker sends no chunk for this long, since Jigasi stops sending during silence; `0` turns the timer off |
| `ASR_UTTERANCE_END_SILENCE_MS` | `0` | caption websocket, utterance mode with `ASR_VAD`: a chunk that ends with at least this much non speech (32 ms windows below `ASR_VAD_THRESHOLD`) closes its speaker's utterance at once, once the utterance holds `ASR_UTTERANCE_MIN_SPEECH_MS` of speech, instead of waiting for the next chunk or `ASR_UTTERANCE_IDLE_S`; `0` turns it off |
| `ASR_UTTERANCE_MIN_SPEECH_MS` | `500` | caption websocket: an utterance with less speech than this (32 ms voice windows with `ASR_VAD`, else 20 ms frames at or above `ASR_SILENCE_RMS`) is dropped instead of transcribed alone; its caption is still sent |
| `ASR_UTTERANCE_FINAL_CAPTION` | `true` | caption websocket, utterance mode only: each 2 s caption is sent as interim under the utterance's message id with the utterance's captions so far, and the utterance text is sent once as the final caption under the same id; a dropped or failed utterance ends with its captions. `false` keeps every 2 s caption final |
| `ASR_VAD` | unset, follows `ASR_UTTERANCE_MODE` | caption websocket: of the chunks that pass the RMS gate, only those Silero VAD hears voice in reach the model, so a fan, a breath or another speaker's faint echo never becomes a caption |
| `ASR_VAD_THRESHOLD` | `0.5` | caption websocket: speech probability at or above which a 32 ms window counts as voice |
| `ASR_VAD_MIN_SPEECH_MS` | `96` | caption websocket: voice a chunk needs, in ms of such windows, to reach the model and to join an utterance |
| `ASR_OUTPUT_GUARD` | unset, follows `ASR_UTTERANCE_MODE` | caption websocket: a caption or utterance text that is a model reply rather than speech ("I'm sorry", "could you please repeat", "the audio is silent", a lone "Okay.") is neither sent nor stored; a dropped utterance ends with its remaining captions |
| `LOG_LEVEL` | `INFO` | application and uvicorn log level |

Profiles shipped in `model_profiles.yml`: `speechlmm` (plain text prompt plus WAV),
`qwen25_omni` and `voxtral_mini_3b_2507` (JSON envelope). A profile overrides only the Triton
fields it lists.

## Container

```sh
docker build \
  --build-arg VERSION=$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2) \
  --build-arg REVISION=$(git rev-parse HEAD) \
  -t fastapi-server .
docker run --rm -p 8000:8000 -e MONGO_URI=mongodb://<host>:27017 -e JWT_SECRET_KEY=<secret> \
  -v /path/to/recordings:/mnt/recordings fastapi-server
```

The image runs as uid and gid 1000, listens on `HOST` and `PORT` (`0.0.0.0:8000`) and reads
every setting above from the environment; no `.env` file is copied in. `/health`
reports the version from `config.py` and, as `commit`, the `REVISION` the image was built from
(`unknown` without it). The
default `RECORDINGS_BASE_PATH`, `/mnt/recordings`, exists in the image and belongs to uid 1000;
a directory mounted there or anywhere else must be writable by that uid.

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
| GET | `/`, `/health` | liveness; `/health` reports the running version and commit |

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
