"""
Store Intelligence System - Event Ingestion Script
Reads generated JSONL events and POSTs them to the API.
"""

import json
import sys
import os
import requests
from pathlib import Path


def ingest_events(events_dir: str, api_url: str = "http://localhost:8000"):
    """Read all JSONL files and POST events to the API in batches."""
    events_path = Path(events_dir)

    if not events_path.exists():
        print(f"Events directory not found: {events_dir}")
        sys.exit(1)

    jsonl_files = list(events_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {events_dir}")
        sys.exit(1)

    total_accepted = 0
    total_rejected = 0

    for jsonl_file in jsonl_files:
        print(f"\n📄 Processing: {jsonl_file.name}")

        events = []
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        print(f"   Found {len(events)} events")

        # Send in batches of 100
        batch_size = 100
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]

            try:
                resp = requests.post(
                    f"{api_url}/events/ingest",
                    json={"events": batch},
                    timeout=30,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    total_accepted += data["accepted"]
                    total_rejected += data["rejected"]
                    print(f"   Batch {i // batch_size + 1}: "
                          f"✅ {data['accepted']} accepted, "
                          f"❌ {data['rejected']} rejected")
                else:
                    print(f"   Batch {i // batch_size + 1}: "
                          f"⚠️ HTTP {resp.status_code}: {resp.text[:200]}")

            except requests.exceptions.ConnectionError:
                print(f"   ❌ Cannot connect to API at {api_url}. Is it running?")
                sys.exit(1)

    print(f"\n{'='*50}")
    print(f"📊 Ingestion Complete!")
    print(f"   ✅ Total Accepted: {total_accepted}")
    print(f"   ❌ Total Rejected: {total_rejected}")
    print(f"{'='*50}")


if __name__ == "__main__":
    events_dir = sys.argv[1] if len(sys.argv) > 1 else "./output/events"
    api_url = sys.argv[2] if len(sys.argv) > 2 else os.getenv("API_URL", "http://localhost:8000")
    ingest_events(events_dir, api_url)
