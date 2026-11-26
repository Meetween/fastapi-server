from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent / ".env"
_REPO_ROOT = Path(__file__).resolve().parent

_DEFAULT_TRITON_PROFILE: dict[str, str] = {
    "url": "http://localhost:9100",
    "model_name": "speechlmm",
    "prompt_protocol": "speechlmm_plain",
}
_PROFILE_TRITON_KEYS = frozenset({"url", "model_name", "prompt_protocol", "caption_chars"})


class TritonSettings(BaseModel):
    """Triton inference client settings."""

    model_config = ConfigDict(frozen=True)
    url: str
    model_name: str
    timeout: int
    max_context_chars: int
    max_history_turns: int
    max_history_chars: int
    prompt_protocol: str
    caption_chars: int


class AppJwtSettings(BaseModel):
    """Application login JWT (access / refresh)."""

    model_config = ConfigDict(frozen=True)
    secret_key: str
    algorithm: str
    access_expire_minutes: int
    refresh_expire_days: int
    audience_access: str
    audience_refresh: str


class MeetingJwtSettings(BaseModel):
    """Jitsi Prosody meeting JWT (`token_auth`). Signed with the same secret as app JWT."""

    model_config = ConfigDict(frozen=True)
    issuer: str
    expire_minutes: int
    meet_base_url: Optional[str]


class MongoSettings(BaseModel):
    """MongoDB connection and collection names."""

    model_config = ConfigDict(frozen=True)
    uri: str
    db_name: str
    users_collection: str
    meeting_context_collection: str


# Display names that belong to a service, not a person. Jigasi joins a transcription as
# "Transcriber" (TranscriptionGatewaySession.DISPLAY_NAME); Jibri records as "Recorder".
# Matched case insensitively. The default is not empty on purpose: a room presence check that
# fails open is what let a transcriber count as a human.
DEFAULT_VP_SERVICE_NAMES = "Transcriber,Recorder,Jibri,Jigasi"


def _service_name_set(raw: str) -> frozenset[str]:
    return frozenset(name.strip().casefold() for name in raw.split(",") if name.strip())


class VirtualParticipantSettings(BaseModel):
    """Virtual participant control route: vp-worker and Prosody HTTP endpoints."""

    model_config = ConfigDict(frozen=True)
    worker_url: str
    worker_secret: str
    require_human: bool
    max_bots_per_room: int
    prosody_http_url: str
    xmpp_domain: str
    request_timeout_s: float
    service_names: frozenset[str] = Field(
        default_factory=lambda: _service_name_set(DEFAULT_VP_SERVICE_NAMES)
    )


def _csv_tuple(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


class ObjectStoreSettings(BaseModel):
    """S3 compatible object store for Agentar files, pictures and report exports."""

    model_config = ConfigDict(frozen=True)
    backend: Literal["s3", "memory"]
    endpoint_url: str
    region: str
    bucket: str
    prefix: str
    addressing_style: Literal["auto", "path", "virtual"]
    access_key_id: str = Field(repr=False)
    secret_access_key: str = Field(repr=False)

    @property
    def configured(self) -> bool:
        """True when the store can be used. An empty bucket or key answers 503 `storage_unavailable`."""
        if self.backend == "memory":
            return True
        return bool(self.bucket and self.access_key_id and self.secret_access_key)


class AgentarSettings(BaseModel):
    """Agentar quotas, caps and lifecycle timing. Defaults follow the Agentar API contract."""

    model_config = ConfigDict(frozen=True)
    file_max_bytes: int
    user_max_files: int
    user_max_bytes: int
    picture_max_bytes: int
    text_max_chars: int
    chunk_tokens: int
    meetings_per_day: int
    schedule_horizon_days: int
    grace_minutes_default: int
    tick_seconds: int
    join_timeout_seconds: int
    voices: tuple[str, ...]
    default_voice: str
    worker_events_base_url: str
    signup_rate_per_ip: int
    signup_rate_per_email: int
    login_fail_limit: int
    login_fail_window_minutes: int
    login_fail_ip_limit: int
    trusted_proxy_ips: tuple[str, ...]
    room_claims_per_day: int
    any_room: bool
    opener_drafts_per_minute: int
    max_live_total: int
    event_max_skew_seconds: int
    event_left_grace_seconds: int


def _resolve_profiles_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _load_triton_profile_overrides(
    *,
    model_profile: str,
    profiles_path: Path,
) -> dict[str, Any]:
    """Load url/model_name/prompt_protocol (and optional caption_chars) from YAML."""
    merged = dict(_DEFAULT_TRITON_PROFILE)
    try:
        path = _resolve_profiles_path(profiles_path)
        if not path.is_file():
            return merged
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        prof = (data.get("profiles") or {}).get(model_profile)
        if not isinstance(prof, dict):
            return merged
        triton = prof.get("triton")
        if not isinstance(triton, dict):
            return merged
        merged.update(
            {
                k: v
                for k, v in triton.items()
                if k in _PROFILE_TRITON_KEYS and v is not None
            }
        )
    except (OSError, yaml.YAMLError):
        pass
    return merged


def _build_triton_settings(
    *,
    profile: dict[str, Any],
    timeout: int,
    max_context_chars: int,
    max_history_turns: int,
    max_history_chars: int,
    caption_chars_default: int,
) -> TritonSettings:
    return TritonSettings(
        url=str(profile["url"]),
        model_name=str(profile["model_name"]),
        prompt_protocol=str(profile["prompt_protocol"]),
        timeout=timeout,
        max_context_chars=max_context_chars,
        max_history_turns=max_history_turns,
        max_history_chars=max_history_chars,
        caption_chars=int(profile.get("caption_chars", caption_chars_default)),
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Meetween FastAPI Server"
    app_version: str = "0.4.1"  # x-release-please-version
    # Commit the image was built from, set by the Dockerfile. "unknown" when run from source.
    app_commit: str = "unknown"
    host: str = "0.0.0.0"
    port: int = 8000

    recordings_base_path: Path = Path("/mnt/recordings")
    recording_format: Optional[str] = "wav"

    min_catchup_duration_seconds: int = 60
    log_level: str = "INFO"

    recording_retention_days: int = 7

    # Meetings with no open websocket and no activity for this long are cleared from memory.
    meeting_idle_timeout_seconds: int = 6 * 60 * 60
    # How long an ended meeting is remembered, so late updates for it are still rejected.
    ended_meeting_retention_seconds: int = 24 * 60 * 60

    enable_auth: bool = Field(default=True, env="ENABLE_AUTH")
    # Shared bearer Jigasi sends. Empty while auth is on rejects those calls.
    jigasi_meetween_api_key: str = Field(default="", env="JIGASI_MEETWEEN_API_KEY")
    # Shared bearer the virtual participant worker sends to /api/chat and to the chat message
    # route, and nowhere else. Kept apart from the Jigasi key so a worker holds no right to end
    # meetings or push participants. Empty while auth is on rejects those calls.
    vp_meetween_api_key: str = Field(default="", env="VP_MEETWEEN_API_KEY")
    # Display name suffix that marks a virtual participant, the worker's VP_BOT_NAME_SUFFIX.
    # Audio of a participant whose name ends with it is not transcribed, so the bot's own
    # answers never enter the transcript. Empty transcribes everyone.
    vp_bot_name_suffix: str = Field(default="", env="VP_BOT_NAME_SUFFIX")

    jwt_secret_key: str = Field(default="change-me-in-production", env="JWT_SECRET_KEY")
    # The JWT_SECRET_KEY the legacy sha256 password digests were made with, kept after that key
    # is rotated. Only verifies those digests, which are replaced with argon2id at the next
    # login; it signs nothing. Empty verifies them with JWT_SECRET_KEY.
    legacy_password_pepper: str = Field(default="", env="LEGACY_PASSWORD_PEPPER")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(
        default=60, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS"
    )
    jwt_audience_access: str = Field(default="meetween-access", env="JWT_AUDIENCE_ACCESS")
    jwt_audience_refresh: str = Field(default="meetween-refresh", env="JWT_AUDIENCE_REFRESH")
    jwt_jitsi_issuer: str = Field(default="meetween", env="JWT_JITSI_ISSUER")
    jwt_jitsi_expire_minutes: int = Field(default=15, env="JWT_JITSI_EXPIRE_MINUTES")
    jitsi_meet_base_url: Optional[str] = Field(default=None, env="JITSI_MEET_BASE_URL")

    # Agentar feature flags, served by GET /api/auth/config. Off by default so an upgrade never
    # opens registration; the demo env file turns SIGNUP_ENABLED on.
    signup_enabled: bool = Field(default=False, env="SIGNUP_ENABLED")
    email_enabled: bool = Field(default=False, env="EMAIL_ENABLED")
    calendar_enabled: bool = Field(default=False, env="CALENDAR_ENABLED")
    # This deployment runs Agentar. Routes are always registered and each answers 503 for a missing
    # dependency; this switch makes startup refuse a configuration that cannot work (empty worker key).
    agentar_enabled: bool = Field(default=False, env="AGENTAR_ENABLED")
    # Sign up attempts per hour allowed by the throttle.
    signup_rate_per_ip: int = Field(default=5, ge=1, env="SIGNUP_RATE_PER_IP")
    signup_rate_per_email: int = Field(default=3, ge=1, env="SIGNUP_RATE_PER_EMAIL")
    # Failed logins: LOGIN_FAIL_LIMIT per window per client address and per email, and
    # LOGIN_FAIL_IP_LIMIT per address over all emails.
    login_fail_limit: int = Field(default=10, ge=1, env="LOGIN_FAIL_LIMIT")
    login_fail_window_minutes: int = Field(default=15, ge=1, env="LOGIN_FAIL_WINDOW_MINUTES")
    login_fail_ip_limit: int = Field(default=50, ge=1, env="LOGIN_FAIL_IP_LIMIT")
    # Comma separated addresses of the reverse proxy. X-Real-IP is the client address only when the
    # TCP peer is one of them, otherwise the peer is. Never the leftmost X-Forwarded-For.
    trusted_proxy_ips: str = Field(default="", env="TRUSTED_PROXY_IPS")
    # First claims of a new room per user and UTC day.
    room_claims_per_day: int = Field(default=20, ge=1, env="ROOM_CLAIMS_PER_DAY")
    # Demo only: an Agentar may be sent into a room another user holds. Off by default; an invite
    # or host consent flow replaces it.
    agentar_any_room: bool = Field(default=False, env="AGENTAR_ANY_ROOM")
    # Opener drafts per owner per minute: each one is a model call on the shared SpeechLMM.
    agentar_opener_drafts_per_minute: int = Field(default=6, ge=1, env="AGENTAR_OPENER_DRAFTS_PER_MINUTE")

    # Signs Jitsi room tokens and is shared with Prosody as JWT_APP_SECRET. Never falls back to
    # JWT_SECRET_KEY: empty means the token routes answer 503 `jitsi_key_missing`.
    jwt_jitsi_secret_key: str = Field(default="", env="JWT_JITSI_SECRET_KEY")

    # Object store. The Agentar routes answer 503 `storage_unavailable` until the bucket and both
    # keys are set. An empty endpoint means AWS. "memory" is for tests only.
    agentar_object_store: Literal["s3", "memory"] = Field(default="s3", env="AGENTAR_OBJECT_STORE")
    s3_endpoint_url: str = Field(default="", env="S3_ENDPOINT_URL")
    s3_region: str = Field(default="us-east-1", env="S3_REGION")
    s3_bucket: str = Field(default="", env="S3_BUCKET")
    s3_prefix: str = Field(default="", env="S3_PREFIX")
    s3_addressing_style: Literal["auto", "path", "virtual"] = Field(
        default="path", env="S3_ADDRESSING_STYLE"
    )
    s3_access_key_id: str = Field(default="", env="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str = Field(default="", env="S3_SECRET_ACCESS_KEY")

    # Agentar quotas and caps per user. Text per file and chunk size are in characters and tokens.
    agentar_file_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1, env="AGENTAR_FILE_MAX_BYTES")
    agentar_user_max_files: int = Field(default=20, ge=1, env="AGENTAR_USER_MAX_FILES")
    agentar_user_max_bytes: int = Field(default=100 * 1024 * 1024, ge=1, env="AGENTAR_USER_MAX_BYTES")
    agentar_picture_max_bytes: int = Field(default=5 * 1024 * 1024, ge=1, env="AGENTAR_PICTURE_MAX_BYTES")
    agentar_text_max_chars: int = Field(default=200_000, ge=1, env="AGENTAR_TEXT_MAX_CHARS")
    agentar_chunk_tokens: int = Field(default=150, ge=1, env="AGENTAR_CHUNK_TOKENS")
    agentar_meetings_per_day: int = Field(default=10, ge=1, env="AGENTAR_MEETINGS_PER_DAY")
    agentar_schedule_horizon_days: int = Field(default=30, ge=1, env="AGENTAR_SCHEDULE_HORIZON_DAYS")
    # Agentars live at once across all users. The demo GPU renders one Ditto session.
    agentar_max_live_total: int = Field(default=1, ge=1, env="AGENTAR_MAX_LIVE_TOTAL")
    # Worker event checks: `at` may differ from server time by this much, and a `left` event is
    # still accepted this long after the meeting ended.
    agentar_event_max_skew_seconds: int = Field(default=600, ge=1, env="AGENTAR_EVENT_MAX_SKEW_SECONDS")
    agentar_event_left_grace_seconds: int = Field(default=300, ge=0, env="AGENTAR_EVENT_LEFT_GRACE_SECONDS")
    # Lifecycle tick: claims due scheduled meetings and reconciles live ones.
    agentar_tick_seconds: int = Field(default=30, ge=1, env="AGENTAR_TICK_SECONDS")
    # A bot still joining this long after dispatch fails with join_timeout. The worker gives up on
    # its own after VP_JOIN_TIMEOUT_S (45 s) plus the Ditto setup (60 s); this is the backstop.
    agentar_join_timeout_seconds: int = Field(default=180, ge=30, env="AGENTAR_JOIN_TIMEOUT_SECONDS")
    # Minutes a waiting meeting stays open after start_at before it is marked missed (1 to 60).
    agentar_grace_minutes_default: int = Field(
        default=15, ge=1, le=60, env="AGENTAR_GRACE_MINUTES_DEFAULT"
    )
    # Kokoro voice ids offered in onboarding, the first is the default. Each id must be in the
    # worker's VP_TTS_VOICES.
    agentar_voices: str = Field(default="af_heart,af_bella,am_fenrir,am_michael", env="AGENTAR_VOICES")
    # How the worker reaches this API, the same value as the worker's VP_MEETWEEN_API_URL. Used to
    # build the events_url sent at dispatch. Empty: dispatch answers 503 `worker_unavailable`.
    # The worker authenticates its events with VP_MEETWEEN_API_KEY, there is no separate key.
    vp_meetween_api_url: str = Field(default="", env="VP_MEETWEEN_API_URL")

    mongo_uri: str = Field(default="mongodb://localhost:27017", env="MONGO_URI")
    mongo_db_name: str = Field(default="meetween", env="MONGO_DB_NAME")
    mongo_users_collection: str = Field(default="users", env="MONGO_USERS_COLLECTION")
    mongo_meeting_context_collection: str = Field(
        default="meeting_context", env="MONGO_MEETING_CONTEXT_COLLECTION"
    )

    model_profile: str = Field(default="speechlmm", env="MODEL_PROFILE")
    model_profiles_path: Path = Field(default=Path("model_profiles.yml"), env="MODEL_PROFILES_PATH")

    triton_timeout: int = Field(default=1800, env="TRITON_TIMEOUT")
    triton_max_context_chars: int = Field(default=300_000, env="TRITON_MAX_CONTEXT_CHARS")
    triton_max_history_turns: int = Field(default=400, env="TRITON_MAX_HISTORY_TURNS")
    triton_max_history_chars: int = Field(default=600_000, env="TRITON_MAX_HISTORY_CHARS")
    triton_caption_chars: int = Field(default=60, env="TRITON_CAPTION_CHARS")
    # Comma separated names the speakers may say, e.g. "Agentar,Meetween". Added to the
    # transcription prompt so the model spells them correctly. Empty leaves the prompt unchanged.
    asr_vocabulary: str = Field(default="", env="ASR_VOCABULARY")

    # Virtual participant control route. Environment names are the upper case field names.
    vp_worker_url: str = "http://vp-worker:8000"
    vp_worker_secret: str = ""
    vp_require_human: bool = True
    vp_max_bots_per_room: int = 2
    vp_prosody_http_url: str = "http://xmpp.meet.jitsi:5280"
    vp_xmpp_domain: str = "meet.jitsi"
    vp_request_timeout_s: float = 10.0
    # The worker's VP_MAX_LIFETIME_S, same value: a bot's room token lives exactly as long as the bot.
    vp_max_lifetime_s: float = Field(default=7200.0, gt=0)
    # Comma separated display names that must not count as a human in the room.
    vp_service_names: str = DEFAULT_VP_SERVICE_NAMES

    # Silence gate on the caption websocket. A chunk with no speech never reaches the model,
    # which otherwise answers silence as a chatbot. The RMS is in raw int16 units (0 to 32767):
    # 150 is about -47 dBFS. 0 turns the gate off.
    asr_silence_rms: float = Field(default=150.0, env="ASR_SILENCE_RMS")
    # Total time at or above that RMS (20 ms frames) a chunk needs to count as speech.
    asr_silence_min_active_ms: int = Field(default=100, env="ASR_SILENCE_MIN_ACTIVE_MS")
    # Utterance mode: speech chunks of a participant are joined until a pause or the cap,
    # transcribed once, and that text is the transcript line. Live captions keep their 2 s pace.
    # Off keeps one transcript line per 2 s caption chunk.
    asr_utterance_mode: bool = Field(default=False, env="ASR_UTTERANCE_MODE")
    # Longest utterance in seconds. Also bounds the buffer: about 32 KB per second per speaker.
    asr_utterance_max_s: float = Field(default=10.0, env="ASR_UTTERANCE_MAX_S")
    # An utterance also closes when its speaker sends nothing for this long. Jigasi drops silence
    # with its own VAD, so chunks stop rather than turn silent. 0 turns the timer off.
    asr_utterance_idle_s: float = Field(default=3.0, env="ASR_UTTERANCE_IDLE_S")
    # A chunk that ends with at least this much non speech (Silero VAD windows) closes its speaker's
    # utterance at once, without waiting for the next chunk or the idle timer. 0 turns it off. Needs ASR_VAD.
    asr_utterance_end_silence_ms: int = Field(default=0, env="ASR_UTTERANCE_END_SILENCE_MS")
    # An utterance with less speech than this (VAD windows with ASR_VAD, else frames above
    # ASR_SILENCE_RMS) is dropped, not transcribed: alone, a fragment comes back as a chatbot reply.
    asr_utterance_min_speech_ms: int = Field(default=500, env="ASR_UTTERANCE_MIN_SPEECH_MS")
    # In utterance mode, the 2 s captions go out as interim and the utterance text as the final
    # caption, so the room and the virtual participant read whole sentences. Off keeps every 2 s
    # caption final. Direct translations ride on finals only in Jigasi, so they then come per utterance.
    asr_utterance_final_caption: bool = Field(default=True, env="ASR_UTTERANCE_FINAL_CAPTION")
    # Speech detector: of the chunks loud enough for the RMS gate, Silero VAD decides which hold
    # voice and so reach the model, and how much speech an utterance holds. The RMS gate alone
    # passes a fan, a breath, or the echo of another speaker. Unset follows ASR_UTTERANCE_MODE.
    asr_vad: Optional[bool] = Field(default=None, env="ASR_VAD")
    # Speech probability at or above which a 32 ms window counts as voice.
    asr_vad_threshold: float = Field(default=0.5, env="ASR_VAD_THRESHOLD")
    # Voice a chunk needs, in ms of such windows, to reach the model and to join an utterance.
    asr_vad_min_speech_ms: int = Field(default=96, env="ASR_VAD_MIN_SPEECH_MS")
    # Output guard: a caption or utterance text that is a model reply ("I'm sorry, could you
    # please repeat") rather than speech is neither sent nor stored. Unset follows ASR_UTTERANCE_MODE.
    asr_output_guard: Optional[bool] = Field(default=None, env="ASR_OUTPUT_GUARD")
    # Plausibility cap: a caption or utterance text longer than this many characters per second of
    # its audio, plus the slack, is a model that kept talking, not speech. It is neither sent nor
    # stored. Fast speech is about 20 characters a second. 0 turns the cap off.
    asr_caption_max_chars_per_s: float = Field(default=25.0, ge=0, env="ASR_CAPTION_MAX_CHARS_PER_S")
    asr_caption_chars_slack: int = Field(default=20, ge=0, env="ASR_CAPTION_CHARS_SLACK")

    _triton: TritonSettings = PrivateAttr()
    _app_jwt: AppJwtSettings = PrivateAttr()
    _meeting_jwt: MeetingJwtSettings = PrivateAttr()
    _mongo: MongoSettings = PrivateAttr()
    _virtual_participants: VirtualParticipantSettings = PrivateAttr()
    _object_store: ObjectStoreSettings = PrivateAttr()
    _agentar: AgentarSettings = PrivateAttr()

    def model_post_init(self, __context) -> None:
        profile = _load_triton_profile_overrides(
            model_profile=self.model_profile,
            profiles_path=self.model_profiles_path,
        )
        self._triton = _build_triton_settings(
            profile=profile,
            timeout=self.triton_timeout,
            max_context_chars=self.triton_max_context_chars,
            max_history_turns=self.triton_max_history_turns,
            max_history_chars=self.triton_max_history_chars,
            caption_chars_default=self.triton_caption_chars,
        )
        self._app_jwt = AppJwtSettings(
            secret_key=self.jwt_secret_key,
            algorithm=self.jwt_algorithm,
            access_expire_minutes=self.jwt_access_token_expire_minutes,
            refresh_expire_days=self.jwt_refresh_token_expire_days,
            audience_access=self.jwt_audience_access,
            audience_refresh=self.jwt_audience_refresh,
        )
        self._meeting_jwt = MeetingJwtSettings(
            issuer=self.jwt_jitsi_issuer,
            expire_minutes=self.jwt_jitsi_expire_minutes,
            meet_base_url=self.jitsi_meet_base_url,
        )
        self._mongo = MongoSettings(
            uri=self.mongo_uri,
            db_name=self.mongo_db_name,
            users_collection=self.mongo_users_collection,
            meeting_context_collection=self.mongo_meeting_context_collection,
        )
        self._virtual_participants = VirtualParticipantSettings(
            worker_url=self.vp_worker_url,
            worker_secret=self.vp_worker_secret,
            require_human=self.vp_require_human,
            max_bots_per_room=self.vp_max_bots_per_room,
            prosody_http_url=self.vp_prosody_http_url,
            xmpp_domain=self.vp_xmpp_domain,
            request_timeout_s=self.vp_request_timeout_s,
            service_names=_service_name_set(self.vp_service_names),
        )
        self._object_store = ObjectStoreSettings(
            backend=self.agentar_object_store,
            endpoint_url=self.s3_endpoint_url,
            region=self.s3_region,
            bucket=self.s3_bucket,
            prefix=self.s3_prefix.strip("/"),
            addressing_style=self.s3_addressing_style,
            access_key_id=self.s3_access_key_id,
            secret_access_key=self.s3_secret_access_key,
        )
        voices = _csv_tuple(self.agentar_voices)
        if not voices:
            raise ValueError("AGENTAR_VOICES must list at least one voice id")
        self._agentar = AgentarSettings(
            file_max_bytes=self.agentar_file_max_bytes,
            user_max_files=self.agentar_user_max_files,
            user_max_bytes=self.agentar_user_max_bytes,
            picture_max_bytes=self.agentar_picture_max_bytes,
            text_max_chars=self.agentar_text_max_chars,
            chunk_tokens=self.agentar_chunk_tokens,
            meetings_per_day=self.agentar_meetings_per_day,
            schedule_horizon_days=self.agentar_schedule_horizon_days,
            grace_minutes_default=self.agentar_grace_minutes_default,
            tick_seconds=self.agentar_tick_seconds,
            join_timeout_seconds=self.agentar_join_timeout_seconds,
            voices=voices,
            default_voice=voices[0],
            worker_events_base_url=self.vp_meetween_api_url.rstrip("/"),
            signup_rate_per_ip=self.signup_rate_per_ip,
            signup_rate_per_email=self.signup_rate_per_email,
            login_fail_limit=self.login_fail_limit,
            login_fail_window_minutes=self.login_fail_window_minutes,
            login_fail_ip_limit=self.login_fail_ip_limit,
            trusted_proxy_ips=_csv_tuple(self.trusted_proxy_ips),
            room_claims_per_day=self.room_claims_per_day,
            any_room=self.agentar_any_room,
            opener_drafts_per_minute=self.agentar_opener_drafts_per_minute,
            max_live_total=self.agentar_max_live_total,
            event_max_skew_seconds=self.agentar_event_max_skew_seconds,
            event_left_grace_seconds=self.agentar_event_left_grace_seconds,
        )

    @property
    def asr_vad_enabled(self) -> bool:
        return self.asr_utterance_mode if self.asr_vad is None else self.asr_vad

    @property
    def asr_utterance_end_silence_enabled(self) -> bool:
        return self.asr_utterance_mode and self.asr_vad_enabled and self.asr_utterance_end_silence_ms > 0

    @property
    def asr_output_guard_enabled(self) -> bool:
        return self.asr_utterance_mode if self.asr_output_guard is None else self.asr_output_guard

    @property
    def app_jwt(self) -> AppJwtSettings:
        return self._app_jwt

    @property
    def meeting_jwt(self) -> MeetingJwtSettings:
        return self._meeting_jwt

    @property
    def triton(self) -> TritonSettings:
        return self._triton

    @property
    def mongo(self) -> MongoSettings:
        return self._mongo

    @property
    def virtual_participants(self) -> VirtualParticipantSettings:
        return self._virtual_participants

    @property
    def object_store(self) -> ObjectStoreSettings:
        return self._object_store

    @property
    def agentar(self) -> AgentarSettings:
        return self._agentar


settings = Settings()
