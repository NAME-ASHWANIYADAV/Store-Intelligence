"""
Store Intelligence System - Event Emitter
Generates structured JSONL events from pipeline outputs, validated against Pydantic schema.
"""

import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger("event_emitter")


class EventEmitter:
    """
    Generates and writes validated JSONL events from detection pipeline output.
    Each event gets a UUID v4 and ISO-8601 timestamp calculated from frame position.
    """

    def __init__(self, output_dir: str, video_start_time: Optional[datetime] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_start_time = video_start_time or datetime(
            2026, 3, 3, 10, 0, 0, tzinfo=timezone.utc
        )
        self.session_seq: Dict[str, int] = {}  # visitor_id → sequence counter
        self.events: List[Dict] = []

    def _next_seq(self, visitor_id: str) -> int:
        """Get next session sequence number for a visitor."""
        self.session_seq[visitor_id] = self.session_seq.get(visitor_id, 0) + 1
        return self.session_seq[visitor_id]

    def _make_timestamp(self, timestamp_sec: float) -> str:
        """Convert frame timestamp (seconds) to ISO-8601 UTC."""
        ts = self.video_start_time + timedelta(seconds=timestamp_sec)
        return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"

    def emit_entry(
        self,
        store_id: str,
        camera_id: str,
        visitor_id: str,
        timestamp_sec: float,
        confidence: float,
        is_staff: bool = False,
    ):
        """Emit an ENTRY event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "ENTRY",
            "timestamp": self._make_timestamp(timestamp_sec),
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": is_staff,
            "confidence": round(confidence, 4),
            "metadata": {"session_seq": self._next_seq(visitor_id)},
        }
        self.events.append(event)

    def emit_exit(
        self,
        store_id: str,
        camera_id: str,
        visitor_id: str,
        timestamp_sec: float,
        confidence: float,
        is_staff: bool = False,
    ):
        """Emit an EXIT event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "EXIT",
            "timestamp": self._make_timestamp(timestamp_sec),
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": is_staff,
            "confidence": round(confidence, 4),
            "metadata": {"session_seq": self._next_seq(visitor_id)},
        }
        self.events.append(event)

    def emit_zone_enter(
        self,
        store_id: str,
        camera_id: str,
        visitor_id: str,
        zone_id: str,
        timestamp_sec: float,
        confidence: float,
        is_staff: bool = False,
    ):
        """Emit a ZONE_ENTER event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "ZONE_ENTER",
            "timestamp": self._make_timestamp(timestamp_sec),
            "zone_id": zone_id,
            "dwell_ms": 0,
            "is_staff": is_staff,
            "confidence": round(confidence, 4),
            "metadata": {"session_seq": self._next_seq(visitor_id)},
        }
        self.events.append(event)

    def emit_zone_exit(
        self,
        store_id: str,
        camera_id: str,
        visitor_id: str,
        zone_id: str,
        timestamp_sec: float,
        dwell_ms: int,
        confidence: float,
        is_staff: bool = False,
    ):
        """Emit a ZONE_EXIT event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "ZONE_EXIT",
            "timestamp": self._make_timestamp(timestamp_sec),
            "zone_id": zone_id,
            "dwell_ms": dwell_ms,
            "is_staff": is_staff,
            "confidence": round(confidence, 4),
            "metadata": {"session_seq": self._next_seq(visitor_id)},
        }
        self.events.append(event)

    def emit_zone_dwell(
        self,
        store_id: str,
        camera_id: str,
        visitor_id: str,
        zone_id: str,
        timestamp_sec: float,
        dwell_ms: int,
        confidence: float,
        is_staff: bool = False,
    ):
        """Emit a ZONE_DWELL event (periodic, every 30s)."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "ZONE_DWELL",
            "timestamp": self._make_timestamp(timestamp_sec),
            "zone_id": zone_id,
            "dwell_ms": dwell_ms,
            "is_staff": is_staff,
            "confidence": round(confidence, 4),
            "metadata": {"session_seq": self._next_seq(visitor_id)},
        }
        self.events.append(event)

    def emit_billing_queue_join(
        self,
        store_id: str,
        camera_id: str,
        visitor_id: str,
        timestamp_sec: float,
        confidence: float,
        queue_depth: int = 0,
        is_staff: bool = False,
    ):
        """Emit a BILLING_QUEUE_JOIN event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "BILLING_QUEUE_JOIN",
            "timestamp": self._make_timestamp(timestamp_sec),
            "zone_id": "BILLING",
            "dwell_ms": 0,
            "is_staff": is_staff,
            "confidence": round(confidence, 4),
            "metadata": {
                "session_seq": self._next_seq(visitor_id),
                "queue_depth": queue_depth,
            },
        }
        self.events.append(event)

    def emit_reentry(
        self,
        store_id: str,
        camera_id: str,
        visitor_id: str,
        timestamp_sec: float,
        confidence: float,
        is_staff: bool = False,
    ):
        """Emit a REENTRY event (same visitor returned)."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "REENTRY",
            "timestamp": self._make_timestamp(timestamp_sec),
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": is_staff,
            "confidence": round(confidence, 4),
            "metadata": {"session_seq": self._next_seq(visitor_id)},
        }
        self.events.append(event)

    def write_events(self, store_id: str):
        """Write all accumulated events to JSONL file for a store."""
        output_path = self.output_dir / f"{store_id}.jsonl"

        # Sort events by timestamp
        self.events.sort(key=lambda e: e["timestamp"])

        with open(output_path, "a") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")

        logger.info(
            "events_written",
            store_id=store_id,
            event_count=len(self.events),
            output_path=str(output_path),
        )

        written = len(self.events)
        self.events = []
        return written

    def get_event_count(self) -> int:
        return len(self.events)
