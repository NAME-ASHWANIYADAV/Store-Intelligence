"""
Store Intelligence System - Metrics Engine
Computes real-time store analytics: visitors, conversion, dwell, queue, abandonment.
"""

from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, distinct, case

from app.models import Event, VisitorSession
from app.schemas import StoreMetrics


async def compute_metrics(db: AsyncSession, store_id: str) -> StoreMetrics:
    """
    Compute real-time metrics for a store.
    Excludes is_staff=true events. Handles zero-purchase stores.
    """
    now = datetime.now(timezone.utc)

    # 1. Unique visitors (non-staff sessions)
    visitor_result = await db.execute(
        select(func.count(distinct(VisitorSession.visitor_id)))
        .where(VisitorSession.store_id == store_id)
        .where(VisitorSession.is_staff == False)
    )
    unique_visitors = visitor_result.scalar() or 0

    # 2. Conversion rate (visitors who purchased / total visitors)
    converted_result = await db.execute(
        select(func.count(distinct(VisitorSession.visitor_id)))
        .where(VisitorSession.store_id == store_id)
        .where(VisitorSession.is_staff == False)
        .where(VisitorSession.is_converted == True)
    )
    converted_count = converted_result.scalar() or 0
    conversion_rate = (converted_count / unique_visitors) if unique_visitors > 0 else 0.0

    # 3. Average dwell per zone
    zone_dwell_result = await db.execute(
        select(
            Event.zone_id,
            func.avg(Event.dwell_ms).label("avg_dwell")
        )
        .where(Event.store_id == store_id)
        .where(Event.is_staff == False)
        .where(Event.event_type == "ZONE_DWELL")
        .where(Event.zone_id.isnot(None))
        .group_by(Event.zone_id)
    )
    avg_dwell_per_zone = {}
    for row in zone_dwell_result:
        avg_dwell_per_zone[row.zone_id] = round(float(row.avg_dwell), 1)

    # 4. Current queue depth (people in billing zone right now)
    # Count visitors who have BILLING_QUEUE_JOIN but no matching EXIT or BILLING_QUEUE_ABANDON
    queue_result = await db.execute(
        select(func.count(distinct(Event.visitor_id)))
        .where(Event.store_id == store_id)
        .where(Event.is_staff == False)
        .where(Event.event_type == "BILLING_QUEUE_JOIN")
        .where(Event.timestamp >= now - timedelta(minutes=10))
    )
    current_queue_depth = queue_result.scalar() or 0

    # Subtract those who abandoned or exited
    abandon_result = await db.execute(
        select(func.count(distinct(Event.visitor_id)))
        .where(Event.store_id == store_id)
        .where(Event.event_type.in_(["BILLING_QUEUE_ABANDON", "EXIT"]))
        .where(Event.timestamp >= now - timedelta(minutes=10))
    )
    left_queue = abandon_result.scalar() or 0
    current_queue_depth = max(0, current_queue_depth - left_queue)

    # 5. Abandonment rate (abandoned / total queue joins)
    total_joins_result = await db.execute(
        select(func.count(distinct(Event.visitor_id)))
        .where(Event.store_id == store_id)
        .where(Event.is_staff == False)
        .where(Event.event_type == "BILLING_QUEUE_JOIN")
    )
    total_joins = total_joins_result.scalar() or 0

    abandon_count_result = await db.execute(
        select(func.count(distinct(Event.visitor_id)))
        .where(Event.store_id == store_id)
        .where(Event.is_staff == False)
        .where(Event.event_type == "BILLING_QUEUE_ABANDON")
    )
    total_abandons = abandon_count_result.scalar() or 0
    abandonment_rate = (total_abandons / total_joins) if total_joins > 0 else 0.0

    return StoreMetrics(
        store_id=store_id,
        unique_visitors=unique_visitors,
        conversion_rate=round(conversion_rate, 4),
        avg_dwell_per_zone=avg_dwell_per_zone,
        current_queue_depth=current_queue_depth,
        abandonment_rate=round(abandonment_rate, 4),
        timestamp=now.isoformat(),
    )
