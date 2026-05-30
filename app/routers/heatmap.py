"""
Store Intelligence System - Heatmap Router
GET /stores/{store_id}/heatmap — zone visit heatmap.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import StoreHeatmap
from app.services.heatmap_engine import compute_heatmap

router = APIRouter(prefix="/stores", tags=["analytics"])


@router.get("/{store_id}/heatmap", response_model=StoreHeatmap)
async def get_heatmap(
    store_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get zone heatmap for a store.

    Returns visit_count and avg_dwell per zone, normalised 0-100
    (highest zone = 100). Includes data_confidence flag (LOW if < 20 sessions).
    """
    return await compute_heatmap(db, store_id)
