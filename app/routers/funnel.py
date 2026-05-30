"""
Store Intelligence System - Funnel Router
GET /stores/{store_id}/funnel — conversion funnel analytics.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import StoreFunnel
from app.services.funnel_engine import compute_funnel

router = APIRouter(prefix="/stores", tags=["analytics"])


@router.get("/{store_id}/funnel", response_model=StoreFunnel)
async def get_funnel(
    store_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get conversion funnel for a store.

    Stages: Entry → Zone Visit → Billing Queue → Purchase.
    Unit: unique visitor sessions (not raw events).
    Re-entries are counted as the same visitor (no double-counting).
    """
    return await compute_funnel(db, store_id)
