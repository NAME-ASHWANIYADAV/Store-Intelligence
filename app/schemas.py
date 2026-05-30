"""
Store Intelligence System - Pydantic Schemas
Request/response validation models matching the exact event schema from the problem statement.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# ===== Event Type Catalogue =====
class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class AnomalySeverity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


# ===== Event Schemas =====
class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = None


class EventIn(BaseModel):
    """Single event matching the required output schema from the problem statement."""
    event_id: str = Field(..., description="UUID v4 — globally unique")
    store_id: str = Field(..., description="From store_layout.json")
    camera_id: str = Field(..., description="Which camera produced this event")
    visitor_id: str = Field(..., description="Re-ID token — unique per visit session")
    event_type: EventType = Field(..., description="Event type from catalogue")
    timestamp: str = Field(..., description="ISO-8601 UTC")
    zone_id: Optional[str] = Field(None, description="null for ENTRY/EXIT events")
    dwell_ms: int = Field(0, description="Duration; 0 for instantaneous events")
    is_staff: bool = Field(False, description="Staff classification flag")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Detection confidence")
    metadata: Optional[EventMetadata] = None

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"Invalid ISO-8601 timestamp: {v}")
        return v


class EventBatchIn(BaseModel):
    """Batch of up to 500 events for ingestion."""
    events: List[EventIn] = Field(..., max_length=500)


class EventError(BaseModel):
    event_id: str
    reason: str


class IngestResponse(BaseModel):
    """Response for POST /events/ingest with partial success support."""
    accepted: int
    rejected: int
    errors: List[EventError] = []


# ===== Metrics Schemas =====
class ZoneDwell(BaseModel):
    zone_id: str
    avg_dwell_ms: float
    visit_count: int


class StoreMetrics(BaseModel):
    store_id: str
    unique_visitors: int
    conversion_rate: float
    avg_dwell_per_zone: Dict[str, float]
    current_queue_depth: int
    abandonment_rate: float
    timestamp: str


# ===== Funnel Schemas =====
class FunnelStage(BaseModel):
    name: str
    count: int
    dropoff_pct: float


class StoreFunnel(BaseModel):
    store_id: str
    stages: List[FunnelStage]
    total_sessions: int


# ===== Heatmap Schemas =====
class ZoneHeat(BaseModel):
    zone_id: str
    visit_count: int
    avg_dwell_ms: float
    intensity: float  # 0-100 normalized


class StoreHeatmap(BaseModel):
    store_id: str
    zones: List[ZoneHeat]
    data_confidence: str  # "HIGH" or "LOW"


# ===== Anomaly Schemas =====
class Anomaly(BaseModel):
    type: str
    severity: AnomalySeverity
    current_value: float
    baseline: float
    deviation_sigma: float
    suggested_action: str
    detected_at: str


class StoreAnomalies(BaseModel):
    store_id: str
    anomalies: List[Anomaly]


# ===== Health Schemas =====
class StoreHealth(BaseModel):
    last_event: Optional[str] = None
    status: str  # "ACTIVE" or "STALE_FEED"
    event_count: int = 0


class HealthResponse(BaseModel):
    status: str
    database: str
    redis: str
    stores: Dict[str, StoreHealth]
    uptime_seconds: float
    version: str = "1.0.0"
