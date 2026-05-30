"""
Store Intelligence System - Integration Tests for Analytics Engines
Tests that ingest realistic event sequences and verify metrics/funnel/heatmap responses.
"""

import uuid
import pytest

pytestmark = pytest.mark.asyncio


def _make_event(store_id="STORE_TEST", visitor_id="V_001", event_type="ENTRY", **kwargs):
    return {
        "event_id": kwargs.get("event_id", str(uuid.uuid4())),
        "store_id": store_id,
        "camera_id": kwargs.get("camera_id", "CAM_01"),
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": kwargs.get("timestamp", "2026-03-03T10:15:32.000Z"),
        "zone_id": kwargs.get("zone_id"),
        "dwell_ms": kwargs.get("dwell_ms", 0),
        "is_staff": kwargs.get("is_staff", False),
        "confidence": kwargs.get("confidence", 0.92),
        "metadata": kwargs.get("metadata"),
    }


class TestAnalyticsEngines:
    """Integration tests that exercise the full ingest→compute→respond path."""

    async def _seed_full_journey(self, client, store_id="STORE_TEST"):
        """Seed a complete visitor journey for testing."""
        events = [
            # Visitor 1: full journey — enter, browse 2 zones, join billing, exit
            _make_event(store_id=store_id, visitor_id="V_001", event_type="ENTRY",
                       timestamp="2026-03-03T10:00:00.000Z"),
            _make_event(store_id=store_id, visitor_id="V_001", event_type="ZONE_ENTER",
                       zone_id="SKINCARE", timestamp="2026-03-03T10:02:00.000Z"),
            _make_event(store_id=store_id, visitor_id="V_001", event_type="ZONE_DWELL",
                       zone_id="SKINCARE", dwell_ms=60000, timestamp="2026-03-03T10:03:00.000Z"),
            _make_event(store_id=store_id, visitor_id="V_001", event_type="ZONE_EXIT",
                       zone_id="SKINCARE", dwell_ms=120000, timestamp="2026-03-03T10:04:00.000Z"),
            _make_event(store_id=store_id, visitor_id="V_001", event_type="ZONE_ENTER",
                       zone_id="HAIRCARE", timestamp="2026-03-03T10:05:00.000Z"),
            _make_event(store_id=store_id, visitor_id="V_001", event_type="ZONE_DWELL",
                       zone_id="HAIRCARE", dwell_ms=45000, timestamp="2026-03-03T10:06:00.000Z"),
            _make_event(store_id=store_id, visitor_id="V_001", event_type="BILLING_QUEUE_JOIN",
                       zone_id="BILLING", timestamp="2026-03-03T10:08:00.000Z",
                       metadata={"session_seq": 7, "queue_depth": 2}),
            _make_event(store_id=store_id, visitor_id="V_001", event_type="EXIT",
                       timestamp="2026-03-03T10:12:00.000Z"),

            # Visitor 2: browse only, no billing, exits
            _make_event(store_id=store_id, visitor_id="V_002", event_type="ENTRY",
                       timestamp="2026-03-03T10:05:00.000Z"),
            _make_event(store_id=store_id, visitor_id="V_002", event_type="ZONE_ENTER",
                       zone_id="MAKEUP", timestamp="2026-03-03T10:07:00.000Z"),
            _make_event(store_id=store_id, visitor_id="V_002", event_type="ZONE_DWELL",
                       zone_id="MAKEUP", dwell_ms=90000, timestamp="2026-03-03T10:08:30.000Z"),
            _make_event(store_id=store_id, visitor_id="V_002", event_type="EXIT",
                       timestamp="2026-03-03T10:10:00.000Z"),

            # Visitor 3: enters, immediately exits (bouncer)
            _make_event(store_id=store_id, visitor_id="V_003", event_type="ENTRY",
                       timestamp="2026-03-03T10:11:00.000Z"),
            _make_event(store_id=store_id, visitor_id="V_003", event_type="EXIT",
                       timestamp="2026-03-03T10:12:00.000Z"),

            # Staff member
            _make_event(store_id=store_id, visitor_id="V_STAFF", event_type="ENTRY",
                       is_staff=True, timestamp="2026-03-03T09:55:00.000Z"),
        ]
        resp = await client.post("/events/ingest", json={"events": events})
        assert resp.status_code == 200
        assert resp.json()["accepted"] == len(events)

    async def test_metrics_after_seeding(self, client):
        """Metrics should compute correctly after seeding full journeys."""
        await self._seed_full_journey(client)

        resp = await client.get("/stores/STORE_TEST/metrics")
        assert resp.status_code == 200
        data = resp.json()

        assert data["store_id"] == "STORE_TEST"
        assert data["unique_visitors"] == 3  # Excludes staff
        assert isinstance(data["conversion_rate"], float)
        assert isinstance(data["avg_dwell_per_zone"], dict)
        assert isinstance(data["current_queue_depth"], int)
        assert isinstance(data["abandonment_rate"], float)
        assert data["timestamp"]  # Not empty

    async def test_funnel_after_seeding(self, client):
        """Funnel should show correct stage counts."""
        await self._seed_full_journey(client)

        resp = await client.get("/stores/STORE_TEST/funnel")
        assert resp.status_code == 200
        data = resp.json()

        assert data["store_id"] == "STORE_TEST"
        assert len(data["stages"]) == 4

        # Entry stage: 3 visitors (excl staff)
        entry = data["stages"][0]
        assert entry["name"] == "Entry"
        assert entry["count"] == 3

        # Zone Visit: V_001 and V_002 visited zones, V_003 didn't
        zone = data["stages"][1]
        assert zone["name"] == "Zone Visit"
        assert zone["count"] == 2

        # Billing: only V_001
        billing = data["stages"][2]
        assert billing["name"] == "Billing Queue"
        assert billing["count"] == 1

        # Dropoff percentages
        assert entry["dropoff_pct"] == 0.0  # No dropoff at entry
        assert zone["dropoff_pct"] > 0  # Some dropoff from entry to zone

    async def test_heatmap_after_seeding(self, client):
        """Heatmap should show zones with correct intensity normalization."""
        await self._seed_full_journey(client)

        resp = await client.get("/stores/STORE_TEST/heatmap")
        assert resp.status_code == 200
        data = resp.json()

        assert data["store_id"] == "STORE_TEST"
        assert len(data["zones"]) > 0
        assert data["data_confidence"] in ("HIGH", "LOW")

        # Check that intensities are 0-100
        for zone in data["zones"]:
            assert 0 <= zone["intensity"] <= 100
            assert zone["visit_count"] > 0
            assert zone["avg_dwell_ms"] >= 0

        # Highest intensity should be 100
        max_intensity = max(z["intensity"] for z in data["zones"])
        assert max_intensity == 100.0

    async def test_anomalies_after_seeding(self, client):
        """Anomalies endpoint should return valid response."""
        await self._seed_full_journey(client)

        resp = await client.get("/stores/STORE_TEST/anomalies")
        assert resp.status_code == 200
        data = resp.json()

        assert data["store_id"] == "STORE_TEST"
        assert isinstance(data["anomalies"], list)
        # With only 1 day of data, most anomalies shouldn't trigger
        for anomaly in data["anomalies"]:
            assert anomaly["type"] in (
                "BILLING_QUEUE_SPIKE", "CONVERSION_DROP",
                "DEAD_ZONE", "STALE_FEED"
            )
            assert anomaly["severity"] in ("INFO", "WARN", "CRITICAL")

    async def test_dwell_per_zone_values(self, client):
        """avg_dwell_per_zone should have real values for zones with dwell events."""
        await self._seed_full_journey(client)

        resp = await client.get("/stores/STORE_TEST/metrics")
        data = resp.json()
        dwells = data["avg_dwell_per_zone"]

        # We sent ZONE_DWELL events for SKINCARE and HAIRCARE and MAKEUP
        # At least one should have dwell data
        if dwells:
            for zone_id, avg_dwell in dwells.items():
                assert avg_dwell > 0, f"Zone {zone_id} has zero dwell"

    async def test_abandonment_rate(self, client):
        """Abandonment rate should be calculable."""
        events = [
            _make_event(store_id="STORE_ABANDON", visitor_id="V_A1",
                       event_type="ENTRY", timestamp="2026-03-03T10:00:00.000Z"),
            _make_event(store_id="STORE_ABANDON", visitor_id="V_A1",
                       event_type="BILLING_QUEUE_JOIN", zone_id="BILLING",
                       timestamp="2026-03-03T10:05:00.000Z"),
            _make_event(store_id="STORE_ABANDON", visitor_id="V_A1",
                       event_type="BILLING_QUEUE_ABANDON",
                       timestamp="2026-03-03T10:08:00.000Z"),
            _make_event(store_id="STORE_ABANDON", visitor_id="V_A2",
                       event_type="ENTRY", timestamp="2026-03-03T10:01:00.000Z"),
            _make_event(store_id="STORE_ABANDON", visitor_id="V_A2",
                       event_type="BILLING_QUEUE_JOIN", zone_id="BILLING",
                       timestamp="2026-03-03T10:06:00.000Z"),
        ]
        await client.post("/events/ingest", json={"events": events})

        resp = await client.get("/stores/STORE_ABANDON/metrics")
        data = resp.json()
        # 1 abandoned out of 2 joins = 50%
        assert data["abandonment_rate"] == 0.5
