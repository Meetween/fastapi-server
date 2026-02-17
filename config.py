
import os
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional

class TritonSettings(BaseModel):
    url: str = Field(default="http://localhost:9000", env="TRITON_URL")
    model_name: str = Field(default="speechlmm", env="TRITON_MODEL_NAME")
    timeout: int = Field(default=1800, env="TRITON_TIMEOUT")

class Settings(BaseSettings):

    app_name: str = "Meetween FastAPI Server"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000

    triton: TritonSettings = TritonSettings()

    vad_sample_rate: int = 16000
    vad_threshold: float = 0.5

    recordings_base_path: Path = Path("/mnt/recordings")
    recording_format: Optional[str] = "wav"

    min_catchup_duration_seconds: int = 180  
    log_level: str = "INFO"

settings = Settings()
