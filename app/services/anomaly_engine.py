"""
Store Intelligence System - Anomaly Detection Engine
Z-score + moving average based anomaly detection for retail metrics.
Database-agnostic: works on PostgreSQL and SQLite.

Anomaly Types:
- BILLING_QUEUE_SPIKE: queue_depth > mean + 2σ
- CONVERSION_DROP: today's rate < avg - 1σ
- DEAD_ZONE: no zone visits in 30+ minutes
- STALE_FEED: no events from any camera in 10+ minutes
"""

from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct
import math

from app.models import Event
from app.schemas import StoreAnomalies, Anomaly, AnomalySeverity
from app.config import get_settings

settings = get_settings()


def _classify_severity(sigma: float) -> AnomalySeverity:
    """Classify anomaly severity by deviation."""
    if abs(sigma) >= 2.0:
        return AnomalySeverity.CRITICAL
    elif abs(sigma) >= 1.0:
        return AnomalySeverity.WARN
    return AnomalySeverity.INFO


async def compute_anomalies(db: AsyncSession, store_id: str) -> StoreAnomalies:
    """Detect anomalies for a store using statistical methods."""
    now = datetime.now(timezone.utc)
    anomalies = []

    try:
        await _detect_queue_spike(db, store_id, now, anomalies)
    except Exception:
        pass  # Graceful degradation

    try:
        await _detect_conversion_drop(db, store_id, now, anomalies)
    except Exception:
        pass

    try:
        await _detect_dead_zones(db, store_id, now, anomalies)
    except Exception:
        pass

    try:
        await _detect_stale_feed(db, store_id, now, anomalies)
    except Exception:
        pass

    return StoreAnomalies(store_id=store_id, anomalies=anomalies)


async def _detect_queue_spike(
    db: AsyncSession, store_id: str, now: datetime, anomalies: list
):
    """Detect if current queue depth exceeds historical mean + 2σ."""
    week_ago = now - timedelta(days=7)

    # Instead of date_trunc (PostgreSQL-only), use simple counting per day
    # Get total queue joins per day for last 7 days
    daily_counts = []
    for day_offset in range(7):
        day_start = (now - timedelta(days=day_offset + 1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        day_end = day_start + timedelta(days=1)

        result = await db.execute(
            select(func.count(distinct(Event.visitor_id)))
            .where(Event.store_id == store_id)
            .where(Event.event_type == "BILLING_QUEUE_JOIN")
            .where(Event.timestamp >= day_start)
            .where(Event.timestamp < day_end)
        )
        count = result.scalar() or 0
        if count > 0:
            daily_counts.append(count)

    if len(daily_counts) < 3:
        return

    mean_q = sum(daily_counts) / len(daily_counts)
    variance = sum((x - mean_q) ** 2 for x in daily_counts) / len(daily_counts)
    std_q = math.sqrt(variance) if variance > 0 else 1.0

    # Today's queue count
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    current_result = await db.execute(
        select(func.count(distinct(Event.visitor_id)))
        .where(Event.store_id == store_id)
        .where(Event.event_type == "BILLING_QUEUE_JOIN")
        .where(Event.timestamp >= today_start)
    )
    current_q = current_result.scalar() or 0

    if std_q > 0 and current_q > mean_q + settings.anomaly_queue_spike_sigma * std_q:
        sigma = round((current_q - mean_q) / std_q, 2)
        anomalies.append(Anomaly(
            type="BILLING_QUEUE_SPIKE",
            severity=_classify_severity(sigma),
            current_value=float(current_q),
            baseline=round(mean_q, 2),
            deviation_sigma=sigma,
            suggested_action=(
                f"Open additional billing counter. Current queue depth ({current_q}) "
                f"exceeds 7-day average ({mean_q:.1f}) by {sigma:.1f}σ."
            ),
            detected_at=now.isoformat(),
        ))


async def _detect_conversion_drop(
    db: AsyncSession, store_id: str, now: datetime, anomalies: list
):
    """Detect if today's conversion rate dropped below avg - 1σ."""
    # Calculate daily conversion rates for past 7 days
    rates = []
    for day_offset in range(1, 8):
        day_start = (now - timedelta(days=day_offset)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        day_end = day_start + timedelta(days=1)

        # Total entries for that day
        entry_result = await db.execute(
            select(func.count(distinct(Event.visitor_id)))
            .where(Event.store_id == store_id)
            .where(Event.is_staff == False)
            .where(Event.event_type.in_(["ENTRY", "REENTRY"]))
            .where(Event.timestamp >= day_start)
            .where(Event.timestamp < day_end)
        )
        entries = entry_result.scalar() or 0

        # Billing joins for that day
        billing_result = await db.execute(
            select(func.count(distinct(Event.visitor_id)))
            .where(Event.store_id == store_id)
            .where(Event.is_staff == False)
            .where(Event.event_type == "BILLING_QUEUE_JOIN")
            .where(Event.timestamp >= day_start)
            .where(Event.timestamp < day_end)
        )
        billing = billing_result.scalar() or 0

        if entries > 0:
            rates.append(billing / entries)

    if len(rates) < 3:
        return

    mean_r = sum(rates) / len(rates)
    variance = sum((r - mean_r) ** 2 for r in rates) / len(rates)
    std_r = math.sqrt(variance) if variance > 0 else 0.01

    # Today's rate
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_entry_result = await db.execute(
        select(func.count(distinct(Event.visitor_id)))
        .where(Event.store_id == store_id)
        .where(Event.is_staff == False)
        .where(Event.event_type.in_(["ENTRY", "REENTRY"]))
        .where(Event.timestamp >= today_start)
    )
    today_entries = today_entry_result.scalar() or 0

    today_billing_result = await db.execute(
        select(func.count(distinct(Event.visitor_id)))
        .where(Event.store_id == store_id)
        .where(Event.is_staff == False)
        .where(Event.event_type == "BILLING_QUEUE_JOIN")
        .where(Event.timestamp >= today_start)
    )
    today_billing = today_billing_result.scalar() or 0
    today_rate = today_billing / today_entries if today_entries > 0 else 0.0

    if std_r > 0 and today_rate < mean_r - settings.anomaly_conversion_drop_sigma * std_r:
        sigma = round((mean_r - today_rate) / std_r, 2)
        anomalies.append(Anomaly(
            type="CONVERSION_DROP",
            severity=_classify_severity(sigma),
            current_value=round(today_rate, 4),
            baseline=round(mean_r, 4),
            deviation_sigma=sigma,
            suggested_action=(
                f"Today's conversion rate ({today_rate:.1%}) is below 7-day average "
                f"({mean_r:.1%}) by {sigma:.1f}σ. Review zone layout and staff coverage."
            ),
            detected_at=now.isoformat(),
        ))


async def _detect_dead_zones(
    db: AsyncSession, store_id: str, now: datetime, anomalies: list
):
    """Detect zones with no activity in 30+ minutes."""
    threshold = now - timedelta(minutes=settings.anomaly_dead_zone_minutes)

    all_zones_result = await db.execute(
        select(Event.zone_id)
        .where(Event.store_id == store_id)
        .where(Event.zone_id.isnot(None))
        .distinct()
    )
    all_zones = {row.zone_id for row in all_zones_result}

    active_result = await db.execute(
        select(Event.zone_id)
        .where(Event.store_id == store_id)
        .where(Event.zone_id.isnot(None))
        .where(Event.timestamp >= threshold)
        .distinct()
    )
    active_zones = {row.zone_id for row in active_result}

    dead_zones = all_zones - active_zones
    for zone_id in dead_zones:
        anomalies.append(Anomaly(
            type="DEAD_ZONE",
            severity=AnomalySeverity.WARN,
            current_value=0.0,
            baseline=1.0,
            deviation_sigma=2.0,
            suggested_action=(
                f"Zone '{zone_id}' has had no visitor activity for "
                f"{settings.anomaly_dead_zone_minutes}+ minutes. "
                f"Check if zone is accessible and displays are visible."
            ),
            detected_at=now.isoformat(),
        ))


async def _detect_stale_feed(
    db: AsyncSession, store_id: str, now: datetime, anomalies: list
):
    """Detect if no events received from any camera in 10+ minutes."""
    threshold = now - timedelta(minutes=settings.anomaly_stale_feed_minutes)

    latest_result = await db.execute(
        select(func.max(Event.timestamp))
        .where(Event.store_id == store_id)
    )
    latest_event = latest_result.scalar()

    if latest_event and latest_event < threshold:
        minutes_stale = round((now - latest_event).total_seconds() / 60, 1)
        anomalies.append(Anomaly(
            type="STALE_FEED",
            severity=AnomalySeverity.CRITICAL,
            current_value=minutes_stale,
            baseline=float(settings.anomaly_stale_feed_minutes),
            deviation_sigma=3.0,
            suggested_action=(
                f"No events received for {minutes_stale:.0f} minutes. "
                f"Check camera connectivity and pipeline health."
            ),
            detected_at=now.isoformat(),
        ))
