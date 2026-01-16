
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):

    app_name: str = "Meetween FastAPI Server"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000

    triton_url: str = os.getenv("TRITON_URL", "http://localhost:9000")
    triton_model_name: str = os.getenv("TRITON_MODEL_NAME", "speechlmm")
    triton_timeout: int = int(os.getenv("TRITON_TIMEOUT", "1800"))

    vad_sample_rate: int = 16000
    vad_threshold: float = 0.5

    recordings_base_path: Path = Path("/mnt/recordings")
    recording_format: Optional[str] = "wav"

    log_level: str = "INFO"

settings = Settings()
