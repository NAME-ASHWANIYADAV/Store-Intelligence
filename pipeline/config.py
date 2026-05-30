"""
Store Intelligence System - Pipeline Configuration
Centralized configuration for the detection pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PipelineConfig:
    """Configuration for the detection pipeline."""

    # Paths
    data_dir: str = os.getenv("DATA_DIR", r"c:\Users\HP\OneDrive\Desktop\purplle\data")
    output_dir: str = os.getenv("OUTPUT_DIR", "./output/events")
    store_layout_path: str = ""
    pos_transactions_path: str = ""
    clips_dir: str = ""

    # YOLO Detection
    yolo_model: str = os.getenv("YOLO_MODEL", "yolo11l.pt")
    detection_confidence: float = float(os.getenv("DETECTION_CONFIDENCE", "0.25"))
    vid_stride: int = int(os.getenv("VID_STRIDE", "3"))
    person_class_id: int = 0  # COCO person class

    # Tracker
    tracker_config: str = "botsort.yaml"

    # Staff Detection
    staff_dwell_threshold: float = float(os.getenv("STAFF_DWELL_THRESHOLD", "0.6"))
    xai_api_key: str = os.getenv("XAI_API_KEY", "")
    xai_api_url: str = os.getenv("XAI_API_URL", "https://api.x.ai/v1/chat/completions")
    xai_model: str = os.getenv("XAI_MODEL", "grok-2-vision-latest")

    # Re-entry Detection
    reentry_similarity_threshold: float = float(os.getenv("REENTRY_SIMILARITY_THRESHOLD", "0.7"))
    reentry_ttl_minutes: int = int(os.getenv("REENTRY_TTL_MINUTES", "30"))

    # Zone Dwell
    dwell_report_interval_ms: int = 30000  # Report dwell every 30s

    def __post_init__(self):
        self.store_layout_path = os.path.join(self.data_dir, "store_layout.json")
        self.pos_transactions_path = os.path.join(self.data_dir, "pos_transactions.csv")
        self.clips_dir = os.path.join(self.data_dir, "CCTV Footage")
        os.makedirs(self.output_dir, exist_ok=True)


def get_pipeline_config() -> PipelineConfig:
    return PipelineConfig()
