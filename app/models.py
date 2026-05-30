"""
Store Intelligence System - SQLAlchemy ORM Models
Database tables for events and visitor sessions.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Boolean, Integer, DateTime,
    JSON, Index, BigInteger, Text, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base


class Event(Base):
    """Stores all ingested behavioral events from the detection pipeline."""
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), unique=True, nullable=False, index=True)
    store_id = Column(String(32), nullable=False, index=True)
    camera_id = Column(String(32), nullable=False)
    visitor_id = Column(String(32), nullable=False, index=True)
    event_type = Column(String(32), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    zone_id = Column(String(32), nullable=True)
    dwell_ms = Column(Integer, default=0)
    is_staff = Column(Boolean, default=False)
    confidence = Column(Float, default=0.0)
    metadata_json = Column(JSON, nullable=True)

    # Composite indexes for common query patterns
    __table_args__ = (
        Index("ix_events_store_timestamp", "store_id", "timestamp"),
        Index("ix_events_store_visitor", "store_id", "visitor_id"),
        Index("ix_events_store_type", "store_id", "event_type"),
        Index("ix_events_visitor_type", "visitor_id", "event_type"),
    )

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "store_id": self.store_id,
            "camera_id": self.camera_id,
            "visitor_id": self.visitor_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "zone_id": self.zone_id,
            "dwell_ms": self.dwell_ms,
            "is_staff": self.is_staff,
            "confidence": self.confidence,
            "metadata": self.metadata_json,
        }


class VisitorSession(Base):
    """Aggregated visitor session built from events."""
    __tablename__ = "visitor_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    visitor_id = Column(String(32), nullable=False, index=True)
    store_id = Column(String(32), nullable=False, index=True)
    entry_time = Column(DateTime(timezone=True), nullable=True)
    exit_time = Column(DateTime(timezone=True), nullable=True)
    zones_visited = Column(JSON, default=list)
    total_dwell_ms = Column(Integer, default=0)
    is_staff = Column(Boolean, default=False)
    is_converted = Column(Boolean, default=False)
    is_reentry = Column(Boolean, default=False)
    event_count = Column(Integer, default=0)

    __table_args__ = (
        Index("ix_sessions_store_entry", "store_id", "entry_time"),
        UniqueConstraint("visitor_id", "store_id", name="uq_visitor_store"),
    )

    def to_dict(self):
        return {
            "visitor_id": self.visitor_id,
            "store_id": self.store_id,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "zones_visited": self.zones_visited,
            "total_dwell_ms": self.total_dwell_ms,
            "is_staff": self.is_staff,
            "is_converted": self.is_converted,
            "is_reentry": self.is_reentry,
            "event_count": self.event_count,
        }
