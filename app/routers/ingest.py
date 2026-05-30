"""
Store Intelligence System - Ingest Router
POST /events/ingest — idempotent batch event ingestion with partial success.
"""

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import EventBatchIn, IngestResponse
from app.services.ingestion import ingest_events

router = APIRouter(prefix="/events", tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    batch: EventBatchIn,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """
    Ingest a batch of events (up to 500).

    - **Idempotent**: duplicate event_ids are silently accepted (no error).
    - **Partial success**: valid events are stored even if some fail validation.
    - Returns count of accepted and rejected events with error details.
    """
    accepted, rejected, errors = await ingest_events(db, batch.events)

    # Set event count header for logging middleware
    response.headers["X-Event-Count"] = str(len(batch.events))

    return IngestResponse(
        accepted=accepted,
        rejected=rejected,
        errors=errors,
    )
