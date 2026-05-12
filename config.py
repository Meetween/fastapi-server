
import os
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional
import yaml

class TritonSettings(BaseModel):
    url: str = Field(default="http://localhost:9000", env="TRITON_URL")
    model_name: str = Field(default="qwen25_omni", env="TRITON_MODEL_NAME")
    timeout: int = Field(default=1800, env="TRITON_TIMEOUT")
    max_context_chars: int = Field(default=300_000, env="TRITON_MAX_CONTEXT_CHARS")
    max_history_turns: int = Field(default=400, env="TRITON_MAX_HISTORY_TURNS")
    max_history_chars: int = Field(default=600_000, env="TRITON_MAX_HISTORY_CHARS")
    prompt_protocol: str = Field(default="speechlmm_plain", env="TRITON_PROMPT_PROTOCOL")
    caption_chars: int = Field(default=60, env="TRITON_CAPTION_CHARS")

class Settings(BaseSettings):

    app_name: str = "Meetween FastAPI Server"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000

    triton: TritonSettings = TritonSettings()

    model_profile: str = Field(default="speechlmm", env="MODEL_PROFILE")
    model_profiles_path: Path = Field(default=Path("model_profiles.yml"), env="MODEL_PROFILES_PATH")

    recordings_base_path: Path = Path("/mnt/recordings")
    recording_format: Optional[str] = "wav"

    min_catchup_duration_seconds: int = 60
    log_level: str = "INFO"

    recording_retention_days: int = 7

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

    mongo_uri: str = Field(default="mongodb://localhost:27017", env="MONGO_URI")
    mongo_db_name: str = Field(default="meetween", env="MONGO_DB_NAME")
    mongo_users_collection: str = Field(default="users", env="MONGO_USERS_COLLECTION")
    mongo_meeting_context_collection: str = Field(
        default="meeting_context", env="MONGO_MEETING_CONTEXT_COLLECTION"
    )

    def model_post_init(self, __context) -> None:
        """
        Optional profile loader (plug-and-play).
        If MODEL_PROFILE is set and the YAML file exists, override self.triton fields.
        """
        try:
            path = self.model_profiles_path
            if not path.is_absolute():
                # Resolve relative to repo root (this file's directory)
                path = (Path(__file__).parent / path).resolve()
            if not path.exists():
                return
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            profiles = (data or {}).get("profiles") or {}
            prof = profiles.get(self.model_profile)
            if not isinstance(prof, dict):
                return
            triton = prof.get("triton") or {}
            if isinstance(triton, dict):
                # Override only provided keys.
                self.triton = self.triton.model_copy(
                    update={k: v for k, v in triton.items() if v is not None}
                )
        except Exception:
            # Keep defaults/env behavior if profile loading fails.
            return

settings = Settings()
