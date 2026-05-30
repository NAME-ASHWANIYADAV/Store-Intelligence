"""
Store Intelligence System - Health Router
GET /health — system health check with per-store feed status.
"""

import time
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text

from app.database import get_db
from app.models import Event
from app.schemas import HealthResponse, StoreHealth
from app.config import get_settings

router = APIRouter(tags=["health"])

# Track startup time
_startup_time = time.time()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: AsyncSession = Depends(get_db),
):
    """
    System health check.

    Returns:
    - Overall status (healthy/degraded)
    - Database connectivity
    - Redis connectivity
    - Per-store status (ACTIVE if events received within 10 min, else STALE_FEED)
    - Uptime in seconds
    """
    now = datetime.now(timezone.utc)
    uptime = round(time.time() - _startup_time, 2)
    stale_threshold = now - timedelta(minutes=settings.anomaly_stale_feed_minutes)

    # Check database connectivity
    db_status = "connected"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    # Check Redis connectivity
    redis_status = "connected"
    try:
        import redis as redis_lib
        r = redis_lib.from_url(settings.redis_url, socket_timeout=2)
        r.ping()
        r.close()
    except Exception:
        redis_status = "disconnected"

    # Per-store status
    stores = {}
    store_result = await db.execute(
        select(
            Event.store_id,
            func.max(Event.timestamp).label("last_event"),
            func.count(Event.id).label("event_count"),
        )
        .group_by(Event.store_id)
    )

    for row in store_result:
        last_event_ts = row.last_event
        status = "ACTIVE"
        if last_event_ts:
            # Normalize: make both tz-aware or both naive for comparison
            if last_event_ts.tzinfo is None:
                compare_threshold = stale_threshold.replace(tzinfo=None)
            else:
                compare_threshold = stale_threshold
            if last_event_ts < compare_threshold:
                status = "STALE_FEED"

        stores[row.store_id] = StoreHealth(
            last_event=last_event_ts.isoformat() if last_event_ts else None,
            status=status,
            event_count=row.event_count,
        )

    overall = "healthy"
    if db_status != "connected":
        overall = "degraded"

    return HealthResponse(
        status=overall,
        database=db_status,
        redis=redis_status,
        stores=stores,
        uptime_seconds=uptime,
    )
