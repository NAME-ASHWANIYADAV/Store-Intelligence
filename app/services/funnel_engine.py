"""
Store Intelligence System - Funnel Engine
Conversion funnel: Entry → Zone Visit → Billing Queue → Purchase.
Unit is sessions, not raw events. Re-entries don't double-count.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct, and_

from app.models import Event, VisitorSession
from app.schemas import StoreFunnel, FunnelStage


async def compute_funnel(db: AsyncSession, store_id: str) -> StoreFunnel:
    """
    Compute the conversion funnel for a store.
    Stages: Entry → Zone Visit → Billing Queue → Purchase
    Uses distinct visitor_id to prevent re-entry double-counting.
    """

    # Stage 1: Entry — unique visitors who entered (non-staff)
    entry_result = await db.execute(
        select(func.count(distinct(Event.visitor_id)))
        .where(Event.store_id == store_id)
        .where(Event.is_staff == False)
        .where(Event.event_type.in_(["ENTRY", "REENTRY"]))
    )
    entry_count = entry_result.scalar() or 0

    # Stage 2: Zone Visit — unique visitors who visited at least one zone
    zone_result = await db.execute(
        select(func.count(distinct(Event.visitor_id)))
        .where(Event.store_id == store_id)
        .where(Event.is_staff == False)
        .where(Event.event_type.in_(["ZONE_ENTER", "ZONE_DWELL"]))
    )
    zone_count = zone_result.scalar() or 0

    # Stage 3: Billing Queue — unique visitors who joined billing queue
    billing_result = await db.execute(
        select(func.count(distinct(Event.visitor_id)))
        .where(Event.store_id == store_id)
        .where(Event.is_staff == False)
        .where(Event.event_type == "BILLING_QUEUE_JOIN")
    )
    billing_count = billing_result.scalar() or 0

    # Stage 4: Purchase — unique visitors who converted (from sessions)
    purchase_result = await db.execute(
        select(func.count(distinct(VisitorSession.visitor_id)))
        .where(VisitorSession.store_id == store_id)
        .where(VisitorSession.is_staff == False)
        .where(VisitorSession.is_converted == True)
    )
    purchase_count = purchase_result.scalar() or 0

    # Calculate drop-off percentages
    stages = []

    # Entry stage
    stages.append(FunnelStage(
        name="Entry",
        count=entry_count,
        dropoff_pct=0.0
    ))

    # Zone Visit stage
    dropoff_zone = round(
        ((entry_count - zone_count) / entry_count * 100) if entry_count > 0 else 0.0, 1
    )
    stages.append(FunnelStage(
        name="Zone Visit",
        count=zone_count,
        dropoff_pct=dropoff_zone
    ))

    # Billing Queue stage
    dropoff_billing = round(
        ((zone_count - billing_count) / zone_count * 100) if zone_count > 0 else 0.0, 1
    )
    stages.append(FunnelStage(
        name="Billing Queue",
        count=billing_count,
        dropoff_pct=dropoff_billing
    ))

    # Purchase stage
    dropoff_purchase = round(
        ((billing_count - purchase_count) / billing_count * 100) if billing_count > 0 else 0.0, 1
    )
    stages.append(FunnelStage(
        name="Purchase",
        count=purchase_count,
        dropoff_pct=dropoff_purchase
    ))

    return StoreFunnel(
        store_id=store_id,
        stages=stages,
        total_sessions=entry_count,
    )
