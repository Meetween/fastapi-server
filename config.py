from pathlib import Path
from typing import Any, Optional

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
    room_owner_collection: str


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

    enable_auth: bool = Field(default=True, env="ENABLE_AUTH")

    jwt_secret_key: str = Field(default="change-me-in-production", env="JWT_SECRET_KEY")
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

    mongo_uri: str = Field(default="mongodb://localhost:27017", env="MONGO_URI")
    mongo_db_name: str = Field(default="meetween", env="MONGO_DB_NAME")
    mongo_users_collection: str = Field(default="users", env="MONGO_USERS_COLLECTION")
    mongo_meeting_context_collection: str = Field(
        default="meeting_context", env="MONGO_MEETING_CONTEXT_COLLECTION"
    )
    mongo_room_owner_collection: str = Field(
        default="room_owner", env="MONGO_ROOM_OWNER_COLLECTION"
    )

    valkey_url: str = Field(default="redis://localhost:6379/0", env="VALKEY_URL")
    # Shared secret the control route presents to vp-worker on every call.
    vp_worker_shared_secret: str = Field(default="", env="VP_WORKER_SHARED_SECRET")
    vp_worker_dispatch_timeout_seconds: float = Field(
        default=10.0, env="VP_WORKER_DISPATCH_TIMEOUT_SECONDS"
    )
    # A pending bot with no alive key after this long is reaped (doc §6).
    vp_bot_pending_timeout_seconds: int = Field(
        default=60, env="VP_BOT_PENDING_TIMEOUT_SECONDS"
    )
    vp_reaper_interval_seconds: int = Field(default=30, env="VP_REAPER_INTERVAL_SECONDS")

    model_profile: str = Field(default="speechlmm", env="MODEL_PROFILE")
    model_profiles_path: Path = Field(default=Path("model_profiles.yml"), env="MODEL_PROFILES_PATH")

    triton_timeout: int = Field(default=1800, env="TRITON_TIMEOUT")
    triton_max_context_chars: int = Field(default=300_000, env="TRITON_MAX_CONTEXT_CHARS")
    triton_max_history_turns: int = Field(default=400, env="TRITON_MAX_HISTORY_TURNS")
    triton_max_history_chars: int = Field(default=600_000, env="TRITON_MAX_HISTORY_CHARS")
    triton_caption_chars: int = Field(default=60, env="TRITON_CAPTION_CHARS")

    _triton: TritonSettings = PrivateAttr()
    _app_jwt: AppJwtSettings = PrivateAttr()
    _meeting_jwt: MeetingJwtSettings = PrivateAttr()
    _mongo: MongoSettings = PrivateAttr()

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
            room_owner_collection=self.mongo_room_owner_collection,
        )

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


settings = Settings()
