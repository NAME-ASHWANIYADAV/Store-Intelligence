"""
Store Intelligence System - Anomalies Router
GET /stores/{store_id}/anomalies — real-time anomaly detection.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import StoreAnomalies
from app.services.anomaly_engine import compute_anomalies

router = APIRouter(prefix="/stores", tags=["analytics"])


@router.get("/{store_id}/anomalies", response_model=StoreAnomalies)
async def get_anomalies(
    store_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get real-time anomaly alerts for a store.

    Detects: BILLING_QUEUE_SPIKE, CONVERSION_DROP, DEAD_ZONE, STALE_FEED.
    Severity levels: INFO (< 1σ), WARN (1-2σ), CRITICAL (> 2σ).
    Includes suggested_action for each anomaly.
    """
    return await compute_anomalies(db, store_id)
