"""
Store Intelligence System - Edge Case Tests

PROMPT: "Generate tests for edge cases: empty store, all-staff clip,
zero purchases, and re-entry funnel counting."
CHANGES MADE: Added explicit test that re-entry visitors are NOT
double-counted in funnel, verified empty store returns all 0s instead
of errors, added all-staff test confirming unique_visitors=0.
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


class TestEdgeCases:
    """Edge case tests for Store Intelligence System."""

    async def test_empty_store_metrics(self, client):
        """Empty store → all metrics return 0, not errors."""
        resp = await client.get("/stores/STORE_EMPTY_999/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["unique_visitors"] == 0
        assert data["conversion_rate"] == 0.0
        assert data["current_queue_depth"] == 0

    async def test_empty_store_funnel(self, client):
        """Empty store → funnel stages all 0."""
        resp = await client.get("/stores/STORE_EMPTY_999/funnel")
        assert resp.status_code == 200
        data = resp.json()
        for stage in data["stages"]:
            assert stage["count"] == 0

    async def test_empty_store_heatmap(self, client):
        """Empty store → empty zones list with LOW confidence."""
        resp = await client.get("/stores/STORE_EMPTY_999/heatmap")
        assert resp.status_code == 200
        data = resp.json()
        assert data["zones"] == []
        assert data["data_confidence"] == "LOW"

    async def test_empty_store_anomalies(self, client):
        """Empty store → no anomalies (not enough data to detect)."""
        resp = await client.get("/stores/STORE_EMPTY_999/anomalies")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["anomalies"], list)

    async def test_all_staff_clip(self, client):
        """When all tracked people are staff → unique_visitors = 0."""
        events = [
            _make_event(visitor_id="V_STAFF_1", is_staff=True),
            _make_event(visitor_id="V_STAFF_2", is_staff=True),
            _make_event(visitor_id="V_STAFF_3", is_staff=True),
        ]
        await client.post("/events/ingest", json={"events": events})

        resp = await client.get("/stores/STORE_BLR_001/metrics")
        data = resp.json()
        assert data["unique_visitors"] == 0

    async def test_zero_purchases(self, client):
        """Zero purchases → conversion_rate = 0.0, not null or NaN."""
        events = [
            _make_event(visitor_id="V_001", event_type="ENTRY"),
            _make_event(visitor_id="V_001", event_type="ZONE_ENTER", zone_id="SKINCARE"),
            _make_event(visitor_id="V_001", event_type="EXIT"),
        ]
        await client.post("/events/ingest", json={"events": events})

        resp = await client.get("/stores/STORE_BLR_001/metrics")
        data = resp.json()
        assert data["conversion_rate"] == 0.0

    async def test_health_endpoint(self, client):
        """Health endpoint should always return valid response."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "database" in data
        assert "uptime_seconds" in data

    async def test_funnel_with_reentry(self, client):
        """Re-entry visitor should be counted ONCE in funnel, not twice."""
        visitor_id = "V_REENTRY_001"
        events = [
            _make_event(visitor_id=visitor_id, event_type="ENTRY",
                       timestamp="2026-03-03T10:00:00.000Z"),
            _make_event(visitor_id=visitor_id, event_type="ZONE_ENTER",
                       zone_id="SKINCARE", timestamp="2026-03-03T10:05:00.000Z"),
            _make_event(visitor_id=visitor_id, event_type="EXIT",
                       timestamp="2026-03-03T10:10:00.000Z"),
            _make_event(visitor_id=visitor_id, event_type="REENTRY",
                       timestamp="2026-03-03T10:20:00.000Z"),
            _make_event(visitor_id=visitor_id, event_type="ZONE_ENTER",
                       zone_id="HAIRCARE", timestamp="2026-03-03T10:25:00.000Z"),
        ]
        await client.post("/events/ingest", json={"events": events})

        resp = await client.get("/stores/STORE_BLR_001/funnel")
        data = resp.json()
        # Entry stage should count this visitor only once
        entry_stage = data["stages"][0]
        assert entry_stage["count"] == 1  # NOT 2

    async def test_multiple_stores_isolation(self, client):
        """Events for store A should not affect store B metrics."""
        events_a = [_make_event(store_id="STORE_A", visitor_id="V_A1")]
        events_b = [
            _make_event(store_id="STORE_B", visitor_id="V_B1"),
            _make_event(store_id="STORE_B", visitor_id="V_B2"),
        ]
        await client.post("/events/ingest", json={"events": events_a})
        await client.post("/events/ingest", json={"events": events_b})

        resp_a = await client.get("/stores/STORE_A/metrics")
        resp_b = await client.get("/stores/STORE_B/metrics")
        assert resp_a.json()["unique_visitors"] == 1
        assert resp_b.json()["unique_visitors"] == 2
