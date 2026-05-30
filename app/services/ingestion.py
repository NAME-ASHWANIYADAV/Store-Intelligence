"""
Store Intelligence System - Event Ingestion Service
Handles event validation, deduplication, storage, and session building.
"""

from datetime import datetime, timezone
from typing import List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models import Event, VisitorSession
from app.schemas import EventIn, EventError

import structlog

logger = structlog.get_logger("ingestion")


async def ingest_events(
    db: AsyncSession, events: List[EventIn]
) -> Tuple[int, int, List[EventError]]:
    """
    Ingest a batch of events with idempotency and partial success.

    Returns (accepted_count, rejected_count, errors).
    Idempotent by event_id — duplicate events are silently accepted.
    """
    accepted = 0
    rejected = 0
    errors: List[EventError] = []

    for event_data in events:
        try:
            # Parse timestamp
            ts = datetime.fromisoformat(
                event_data.timestamp.replace("Z", "+00:00")
            )

            # Build metadata dict
            meta = None
            if event_data.metadata:
                meta = event_data.metadata.model_dump(exclude_none=True)

            # Upsert with ON CONFLICT DO NOTHING (idempotent)
            stmt = pg_insert(Event).values(
                event_id=event_data.event_id,
                store_id=event_data.store_id,
                camera_id=event_data.camera_id,
                visitor_id=event_data.visitor_id,
                event_type=event_data.event_type.value,
                timestamp=ts,
                zone_id=event_data.zone_id,
                dwell_ms=event_data.dwell_ms,
                is_staff=event_data.is_staff,
                confidence=event_data.confidence,
                metadata_json=meta,
            ).on_conflict_do_nothing(index_elements=["event_id"])

            result = await db.execute(stmt)
            accepted += 1

        except Exception as e:
            rejected += 1
            errors.append(EventError(
                event_id=event_data.event_id,
                reason=str(e)[:200]
            ))
            logger.warning(
                "event_rejected",
                event_id=event_data.event_id,
                reason=str(e)[:200],
            )

    # Commit all accepted events
    if accepted > 0:
        await db.commit()

        # Rebuild sessions for affected visitors
        visitor_ids = list(set(e.visitor_id for e in events))
        store_ids = list(set(e.store_id for e in events))
        await rebuild_sessions(db, visitor_ids, store_ids)

    return accepted, rejected, errors


async def rebuild_sessions(
    db: AsyncSession, visitor_ids: List[str], store_ids: List[str]
):
    """
    Rebuild visitor sessions from events for the given visitors.
    A session is defined by ENTRY → ... → EXIT for a visitor in a store.
    """
    for store_id in store_ids:
        for visitor_id in visitor_ids:
            # Get all events for this visitor in this store, ordered by time
            result = await db.execute(
                select(Event)
                .where(Event.visitor_id == visitor_id)
                .where(Event.store_id == store_id)
                .where(Event.is_staff == False)
                .order_by(Event.timestamp)
            )
            events = result.scalars().all()

            if not events:
                continue

            # Build session
            entry_time = None
            exit_time = None
            zones_visited = set()
            total_dwell = 0
            is_reentry = False
            is_converted = False

            for evt in events:
                if evt.event_type == "ENTRY":
                    if entry_time is None:
                        entry_time = evt.timestamp
                elif evt.event_type == "EXIT":
                    exit_time = evt.timestamp
                elif evt.event_type == "REENTRY":
                    is_reentry = True
                    if entry_time is None:
                        entry_time = evt.timestamp
                elif evt.event_type in ("ZONE_ENTER", "ZONE_DWELL", "ZONE_EXIT"):
                    if evt.zone_id:
                        zones_visited.add(evt.zone_id)
                    if evt.event_type == "ZONE_DWELL":
                        total_dwell += evt.dwell_ms
                elif evt.event_type == "BILLING_QUEUE_JOIN":
                    zones_visited.add("BILLING")

            # Check if any staff event
            staff_result = await db.execute(
                select(Event)
                .where(Event.visitor_id == visitor_id)
                .where(Event.store_id == store_id)
                .where(Event.is_staff == True)
                .limit(1)
            )
            is_staff = staff_result.scalar_one_or_none() is not None

            # Upsert session
            stmt = pg_insert(VisitorSession).values(
                visitor_id=visitor_id,
                store_id=store_id,
                entry_time=entry_time or (events[0].timestamp if events else None),
                exit_time=exit_time,
                zones_visited=list(zones_visited),
                total_dwell_ms=total_dwell,
                is_staff=is_staff,
                is_converted=is_converted,
                is_reentry=is_reentry,
                event_count=len(events),
            ).on_conflict_do_update(
                constraint="uq_visitor_store",
                set_={
                    "entry_time": entry_time or (events[0].timestamp if events else None),
                    "exit_time": exit_time,
                    "zones_visited": list(zones_visited),
                    "total_dwell_ms": total_dwell,
                    "is_staff": is_staff,
                    "is_converted": is_converted,
                    "is_reentry": is_reentry,
                    "event_count": len(events),
                }
            )
            await db.execute(stmt)

        await db.commit()
