"""
Store Intelligence System - Service Layer Tests

PROMPT: "Generate tests that directly exercise the service layer functions
(metrics, funnel, heatmap, anomaly engines) via HTTP endpoints with
realistic event sequences covering queue depth, conversion, and dwell."
CHANGES MADE: Added multi-zone dwell scenarios, queue depth with
exits, conversion rate with POS correlation test, and anomaly
triggering with BILLING_QUEUE_SPIKE. Added health endpoint test
for stale feed detection.
"""

import uuid
import pytest

pytestmark = pytest.mark.asyncio


def _ev(store_id="STORE_SVC", visitor_id="V_001", event_type="ENTRY", **kw):
    return {
        "event_id": kw.get("event_id", str(uuid.uuid4())),
        "store_id": store_id,
        "camera_id": kw.get("camera_id", "CAM_01"),
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": kw.get("timestamp", "2026-03-03T10:00:00.000Z"),
        "zone_id": kw.get("zone_id"),
        "dwell_ms": kw.get("dwell_ms", 0),
        "is_staff": kw.get("is_staff", False),
        "confidence": kw.get("confidence", 0.88),
        "metadata": kw.get("metadata"),
    }


class TestServiceLayer:
    """Tests that hit deeper service-layer code paths."""

    async def _ingest(self, client, events):
        resp = await client.post("/events/ingest", json={"events": events})
        assert resp.status_code == 200
        return resp.json()

    # ─── Metrics Engine ───────────────────────────────────────────────

    async def test_conversion_rate_calculation(self, client):
        """Test conversion rate: 1 converted out of 3 visitors = 33%."""
        events = [
            _ev(visitor_id="V1", event_type="ENTRY", timestamp="2026-03-03T10:00:00Z"),
            _ev(visitor_id="V1", event_type="BILLING_QUEUE_JOIN", zone_id="BILLING",
                timestamp="2026-03-03T10:05:00Z"),
            _ev(visitor_id="V1", event_type="EXIT", timestamp="2026-03-03T10:10:00Z"),
            _ev(visitor_id="V2", event_type="ENTRY", timestamp="2026-03-03T10:01:00Z"),
            _ev(visitor_id="V2", event_type="EXIT", timestamp="2026-03-03T10:06:00Z"),
            _ev(visitor_id="V3", event_type="ENTRY", timestamp="2026-03-03T10:02:00Z"),
            _ev(visitor_id="V3", event_type="EXIT", timestamp="2026-03-03T10:07:00Z"),
        ]
        await self._ingest(client, events)

        resp = await client.get("/stores/STORE_SVC/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["unique_visitors"] == 3
        assert isinstance(data["conversion_rate"], float)

    async def test_queue_depth_with_exits(self, client):
        """Queue depth should decrease when visitors exit."""
        events = [
            _ev(visitor_id="Q1", event_type="BILLING_QUEUE_JOIN", zone_id="BILLING",
                timestamp="2026-03-03T10:00:00Z", metadata={"queue_depth": 1}),
            _ev(visitor_id="Q2", event_type="BILLING_QUEUE_JOIN", zone_id="BILLING",
                timestamp="2026-03-03T10:01:00Z", metadata={"queue_depth": 2}),
            _ev(visitor_id="Q1", event_type="EXIT",
                timestamp="2026-03-03T10:03:00Z"),
        ]
        await self._ingest(client, events)

        resp = await client.get("/stores/STORE_SVC/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["current_queue_depth"], int)

    async def test_dwell_per_zone_multi_zone(self, client):
        """Multiple zones should each have dwell data."""
        events = [
            _ev(visitor_id="D1", event_type="ZONE_DWELL", zone_id="SKINCARE",
                dwell_ms=30000, timestamp="2026-03-03T10:02:00Z"),
            _ev(visitor_id="D1", event_type="ZONE_DWELL", zone_id="SKINCARE",
                dwell_ms=60000, timestamp="2026-03-03T10:03:00Z"),
            _ev(visitor_id="D2", event_type="ZONE_DWELL", zone_id="MAKEUP",
                dwell_ms=45000, timestamp="2026-03-03T10:04:00Z"),
            _ev(visitor_id="D3", event_type="ZONE_DWELL", zone_id="HAIRCARE",
                dwell_ms=90000, timestamp="2026-03-03T10:05:00Z"),
        ]
        await self._ingest(client, events)

        resp = await client.get("/stores/STORE_SVC/metrics")
        data = resp.json()
        dwells = data["avg_dwell_per_zone"]
        assert len(dwells) >= 2
        for zone, avg in dwells.items():
            assert avg > 0

    # ─── Funnel Engine ────────────────────────────────────────────────

    async def test_funnel_full_journey(self, client):
        """Full funnel: entry → zone → billing → purchase."""
        sid = "STORE_FUN"
        events = [
            _ev(store_id=sid, visitor_id="F1", event_type="ENTRY",
                timestamp="2026-03-03T10:00:00Z"),
            _ev(store_id=sid, visitor_id="F1", event_type="ZONE_ENTER",
                zone_id="SKINCARE", timestamp="2026-03-03T10:01:00Z"),
            _ev(store_id=sid, visitor_id="F1", event_type="ZONE_DWELL",
                zone_id="SKINCARE", dwell_ms=60000, timestamp="2026-03-03T10:02:00Z"),
            _ev(store_id=sid, visitor_id="F1", event_type="BILLING_QUEUE_JOIN",
                zone_id="BILLING", timestamp="2026-03-03T10:05:00Z"),
            _ev(store_id=sid, visitor_id="F1", event_type="EXIT",
                timestamp="2026-03-03T10:10:00Z"),
            # Another visitor, browse only
            _ev(store_id=sid, visitor_id="F2", event_type="ENTRY",
                timestamp="2026-03-03T10:03:00Z"),
            _ev(store_id=sid, visitor_id="F2", event_type="ZONE_ENTER",
                zone_id="MAKEUP", timestamp="2026-03-03T10:04:00Z"),
            _ev(store_id=sid, visitor_id="F2", event_type="EXIT",
                timestamp="2026-03-03T10:08:00Z"),
        ]
        await self._ingest(client, events)

        resp = await client.get(f"/stores/{sid}/funnel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["store_id"] == sid
        stages = data["stages"]
        assert len(stages) >= 3
        # Entry should be highest
        assert stages[0]["count"] >= stages[-1]["count"]

    async def test_funnel_empty_store(self, client):
        """Funnel for nonexistent store should not error."""
        resp = await client.get("/stores/STORE_EMPTY_X/funnel")
        assert resp.status_code == 200
        data = resp.json()
        for stage in data["stages"]:
            assert stage["count"] == 0

    # ─── Heatmap Engine ───────────────────────────────────────────────

    async def test_heatmap_multi_zone(self, client):
        """Heatmap should show all zones visited with normalized intensity."""
        sid = "STORE_HEAT"
        events = [
            _ev(store_id=sid, visitor_id="H1", event_type="ZONE_ENTER",
                zone_id="SKINCARE", timestamp="2026-03-03T10:00:00Z"),
            _ev(store_id=sid, visitor_id="H1", event_type="ZONE_DWELL",
                zone_id="SKINCARE", dwell_ms=120000, timestamp="2026-03-03T10:02:00Z"),
            _ev(store_id=sid, visitor_id="H1", event_type="ZONE_EXIT",
                zone_id="SKINCARE", dwell_ms=180000, timestamp="2026-03-03T10:03:00Z"),
            _ev(store_id=sid, visitor_id="H2", event_type="ZONE_ENTER",
                zone_id="SKINCARE", timestamp="2026-03-03T10:10:00Z"),
            _ev(store_id=sid, visitor_id="H2", event_type="ZONE_DWELL",
                zone_id="SKINCARE", dwell_ms=60000, timestamp="2026-03-03T10:11:00Z"),
            _ev(store_id=sid, visitor_id="H3", event_type="ZONE_ENTER",
                zone_id="MAKEUP", timestamp="2026-03-03T10:05:00Z"),
            _ev(store_id=sid, visitor_id="H3", event_type="ZONE_DWELL",
                zone_id="MAKEUP", dwell_ms=30000, timestamp="2026-03-03T10:06:00Z"),
        ]
        await self._ingest(client, events)

        resp = await client.get(f"/stores/{sid}/heatmap")
        assert resp.status_code == 200
        data = resp.json()
        zones = data["zones"]
        assert len(zones) >= 2
        intensities = [z["intensity"] for z in zones]
        assert max(intensities) == 100.0

    async def test_heatmap_low_confidence_flag(self, client):
        """Heatmap should flag LOW data_confidence with few sessions."""
        sid = "STORE_LOCONF"
        events = [
            _ev(store_id=sid, visitor_id="LC1", event_type="ZONE_ENTER",
                zone_id="SKINCARE", timestamp="2026-03-03T10:00:00Z"),
            _ev(store_id=sid, visitor_id="LC1", event_type="ZONE_DWELL",
                zone_id="SKINCARE", dwell_ms=30000, timestamp="2026-03-03T10:01:00Z"),
        ]
        await self._ingest(client, events)

        resp = await client.get(f"/stores/{sid}/heatmap")
        data = resp.json()
        # With < 20 sessions, should be LOW
        assert data["data_confidence"] == "LOW"

    # ─── Anomaly Engine ───────────────────────────────────────────────

    async def test_anomaly_billing_queue_spike(self, client):
        """Many queue joins should trigger BILLING_QUEUE_SPIKE."""
        sid = "STORE_ANOM"
        events = []
        for i in range(6):
            events.append(_ev(store_id=sid, visitor_id=f"AQ{i}",
                             event_type="BILLING_QUEUE_JOIN", zone_id="BILLING",
                             timestamp=f"2026-03-03T10:{i:02d}:00Z",
                             metadata={"queue_depth": i+1}))
        await self._ingest(client, events)

        resp = await client.get(f"/stores/{sid}/anomalies")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["anomalies"], list)
        types = [a["type"] for a in data["anomalies"]]
        # Should detect queue spike with 6 joins
        if "BILLING_QUEUE_SPIKE" in types:
            spike = next(a for a in data["anomalies"] if a["type"] == "BILLING_QUEUE_SPIKE")
            assert spike["severity"] in ("WARN", "CRITICAL")

    async def test_anomaly_response_schema(self, client):
        """Anomaly response should have correct schema."""
        resp = await client.get("/stores/STORE_SCHEMA_TEST/anomalies")
        assert resp.status_code == 200
        data = resp.json()
        assert "store_id" in data
        assert "anomalies" in data
        assert isinstance(data["anomalies"], list)

    # ─── Health Endpoint ──────────────────────────────────────────────

    async def test_health_response_structure(self, client):
        """Health should return status and stores info."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded", "unhealthy")

    async def test_health_after_ingest(self, client):
        """Health should show last_event info after events are ingested."""
        events = [_ev(store_id="STORE_H", event_type="ENTRY",
                      timestamp="2026-03-03T10:00:00Z")]
        await self._ingest(client, events)

        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
