"""
Store Intelligence System - Application Configuration
Centralized settings management via pydantic-settings.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "postgresql+asyncpg://storeintel:storeintel_pass@localhost:5432/store_intelligence"
    database_url_sync: str = "postgresql://storeintel:storeintel_pass@localhost:5432/store_intelligence"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    debug: bool = False

    # Grok Vision API (xAI)
    xai_api_key: str = ""
    xai_api_url: str = "https://api.x.ai/v1/chat/completions"
    xai_model: str = "grok-2-vision-latest"

    # Pipeline
    data_dir: str = "./data"
    output_dir: str = "./output/events"
    yolo_model: str = "yolo11l.pt"
    detection_confidence: float = 0.25
    vid_stride: int = 3
    staff_dwell_threshold: float = 0.6
    reentry_similarity_threshold: float = 0.7
    reentry_ttl_minutes: int = 30

    # Dashboard
    streamlit_port: int = 8501
    api_url: str = "http://localhost:8000"

    # Anomaly detection
    anomaly_queue_spike_sigma: float = 2.0
    anomaly_conversion_drop_sigma: float = 1.0
    anomaly_dead_zone_minutes: int = 30
    anomaly_stale_feed_minutes: int = 10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
