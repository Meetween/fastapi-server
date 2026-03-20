
import os
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional

class TritonSettings(BaseModel):
    url: str = Field(default="http://localhost:9000", env="TRITON_URL")
    model_name: str = Field(default="speechlmm", env="TRITON_MODEL_NAME")
    timeout: int = Field(default=1800, env="TRITON_TIMEOUT")
    max_context_chars: int = Field(default=300_000, env="TRITON_MAX_CONTEXT_CHARS")
    max_history_turns: int = Field(default=400, env="TRITON_MAX_HISTORY_TURNS")
    max_history_chars: int = Field(default=600_000, env="TRITON_MAX_HISTORY_CHARS")

class Settings(BaseSettings):

    app_name: str = "Meetween FastAPI Server"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000

    triton: TritonSettings = TritonSettings()

    recordings_base_path: Path = Path("/mnt/recordings")
    recording_format: Optional[str] = "wav"

    min_catchup_duration_seconds: int = 60
    log_level: str = "INFO"

    recording_retention_days: int = 7

settings = Settings()
