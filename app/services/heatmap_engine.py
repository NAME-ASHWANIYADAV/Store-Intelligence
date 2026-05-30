"""
Store Intelligence System - Heatmap Engine
Zone visit frequency + avg dwell, normalised 0-100, with data_confidence flag.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct

from app.models import Event, VisitorSession
from app.schemas import StoreHeatmap, ZoneHeat


async def compute_heatmap(db: AsyncSession, store_id: str) -> StoreHeatmap:
    """
    Compute zone heatmap for a store.
    Returns visit_count and avg_dwell per zone, normalized 0-100.
    Includes data_confidence flag (LOW if < 20 sessions).
    """

    # Get zone visit counts and avg dwell
    zone_result = await db.execute(
        select(
            Event.zone_id,
            func.count(distinct(Event.visitor_id)).label("visit_count"),
            func.avg(Event.dwell_ms).label("avg_dwell_ms"),
        )
        .where(Event.store_id == store_id)
        .where(Event.is_staff == False)
        .where(Event.zone_id.isnot(None))
        .where(Event.event_type.in_(["ZONE_ENTER", "ZONE_DWELL"]))
        .group_by(Event.zone_id)
    )
    rows = zone_result.all()

    if not rows:
        return StoreHeatmap(
            store_id=store_id,
            zones=[],
            data_confidence="LOW"
        )

    # Find max visit count for normalization
    max_visits = max(row.visit_count for row in rows)

    zones = []
    for row in rows:
        intensity = round((row.visit_count / max_visits) * 100, 1) if max_visits > 0 else 0
        zones.append(ZoneHeat(
            zone_id=row.zone_id,
            visit_count=row.visit_count,
            avg_dwell_ms=round(float(row.avg_dwell_ms or 0), 1),
            intensity=intensity,
        ))

    # Sort by intensity descending
    zones.sort(key=lambda z: z.intensity, reverse=True)

    # Check data confidence
    total_sessions_result = await db.execute(
        select(func.count(distinct(VisitorSession.visitor_id)))
        .where(VisitorSession.store_id == store_id)
        .where(VisitorSession.is_staff == False)
    )
    total_sessions = total_sessions_result.scalar() or 0
    data_confidence = "HIGH" if total_sessions >= 20 else "LOW"

    return StoreHeatmap(
        store_id=store_id,
        zones=zones,
        data_confidence=data_confidence,
    )
