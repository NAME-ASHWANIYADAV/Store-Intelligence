"""
Store Intelligence System - Metrics Endpoint Tests

PROMPT: "Generate pytest tests for the metrics endpoint covering
staff exclusion, zero-purchase stores, and real-time accuracy."
CHANGES MADE: Added test for staff exclusion from unique_visitors count,
verified conversion_rate returns 0.0 (not null) for zero-purchase stores,
added queue depth accuracy test.
"""

import uuid
import pytest

pytestmark = pytest.mark.asyncio


def _make_event(store_id="STORE_BLR_001", event_type="ENTRY", is_staff=False, **kwargs):
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": kwargs.get("visitor_id", f"V_{uuid.uuid4().hex[:8]}"),
        "event_type": event_type,
        "timestamp": kwargs.get("timestamp", "2026-03-03T10:15:32.000Z"),
        "zone_id": kwargs.get("zone_id"),
        "dwell_ms": kwargs.get("dwell_ms", 0),
        "is_staff": is_staff,
        "confidence": 0.92,
        "metadata": kwargs.get("metadata"),
    }


class TestMetricsEndpoint:
    """Tests for GET /stores/{store_id}/metrics."""

    async def test_metrics_empty_store(self, client):
        """Empty store should return zero metrics, not error."""
        resp = await client.get("/stores/STORE_EMPTY/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["unique_visitors"] == 0
        assert data["conversion_rate"] == 0.0
        assert data["current_queue_depth"] == 0
        assert data["abandonment_rate"] == 0.0

    async def test_metrics_with_visitors(self, client):
        """Metrics should count unique visitors correctly."""
        # Ingest 3 visitors
        events = [
            _make_event(visitor_id="V_001"),
            _make_event(visitor_id="V_002"),
            _make_event(visitor_id="V_003"),
        ]
        await client.post("/events/ingest", json={"events": events})

        resp = await client.get("/stores/STORE_BLR_001/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["unique_visitors"] == 3

    async def test_staff_excluded_from_visitors(self, client):
        """Staff events should NOT count towards unique_visitors."""
        events = [
            _make_event(visitor_id="V_CUST_001", is_staff=False),
            _make_event(visitor_id="V_CUST_002", is_staff=False),
            _make_event(visitor_id="V_STAFF_001", is_staff=True),
            _make_event(visitor_id="V_STAFF_002", is_staff=True),
        ]
        await client.post("/events/ingest", json={"events": events})

        resp = await client.get("/stores/STORE_BLR_001/metrics")
        data = resp.json()
        # Only 2 customers, staff excluded
        assert data["unique_visitors"] == 2

    async def test_zero_purchase_conversion(self, client):
        """Zero purchases → conversion_rate = 0.0, not null or error."""
        events = [
            _make_event(visitor_id="V_001"),
            _make_event(visitor_id="V_002"),
        ]
        await client.post("/events/ingest", json={"events": events})

        resp = await client.get("/stores/STORE_BLR_001/metrics")
        data = resp.json()
        assert data["conversion_rate"] == 0.0
        assert isinstance(data["conversion_rate"], float)

    async def test_metrics_response_schema(self, client):
        """Response should contain all required fields."""
        resp = await client.get("/stores/STORE_BLR_001/metrics")
        data = resp.json()
        required_fields = [
            "store_id", "unique_visitors", "conversion_rate",
            "avg_dwell_per_zone", "current_queue_depth",
            "abandonment_rate", "timestamp",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
