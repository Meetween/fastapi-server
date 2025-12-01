
import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    app_name: str = "Meetween FastAPI Server"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000

    speechlmm_model_path: str = os.getenv("SPEECHLMM_MODEL_PATH", "")

    vad_sample_rate: int = 16000
    vad_threshold: float = 0.5

    recordings_base_path: Path = Path("/mnt/recordings")

    log_level: str = "INFO"

settings = Settings()
