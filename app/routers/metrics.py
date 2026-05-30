"""
Store Intelligence System - Metrics Router
GET /stores/{store_id}/metrics — real-time store analytics.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import StoreMetrics
from app.services.metrics_engine import compute_metrics

router = APIRouter(prefix="/stores", tags=["analytics"])


@router.get("/{store_id}/metrics", response_model=StoreMetrics)
async def get_metrics(
    store_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get real-time metrics for a specific store.

    Returns: unique_visitors, conversion_rate, avg_dwell_per_zone,
    current_queue_depth, abandonment_rate.

    Staff events (is_staff=true) are excluded from all calculations.
    Zero-purchase stores return conversion_rate=0.0 (not null or error).
    """
    return await compute_metrics(db, store_id)
