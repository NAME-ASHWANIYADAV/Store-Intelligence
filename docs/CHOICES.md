# Store Intelligence System — Technical Choices

This document captures three significant technical decisions made during development, including what AI tools suggested, what I ultimately chose, and the reasoning behind each override or acceptance.

---

## Decision 1: Detection Model — YOLO11l

### Options Evaluated
| Model | mAP (COCO) | Parameters | Speed | Key Feature |
|-------|-----------|------------|-------|-------------|
| YOLOv8x | ~53.9 | 68.2M | Baseline | Battle-tested, largest community |
| YOLOv8l | ~52.9 | 43.7M | Faster | Smaller but lower accuracy |
| **YOLO11l** | **~53.4** | **~25.3M** | **Faster** | **Native ReID, C2PSA attention** |
| YOLOv12 | Highest | Very high | Slow | Research-grade, unstable |
| RT-DETR | ~54.3 | 42M | Moderate | Transformer-based, NMS-free |

### What AI Suggested
Claude initially recommended YOLOv8x as the "production standard" with the largest community and most tutorials. Gemini suggested considering RT-DETR for its transformer-based architecture.

### What I Chose and Why
I chose **YOLO11l** for three specific reasons:

1. **Native BoT-SORT with ReID**: YOLO11 includes BoT-SORT as a built-in tracker with `with_reid: True` support. This eliminates an entire dependency (separate BoT-SORT or Deep OC-SORT installation), reduces integration bugs, and simplifies the codebase. With YOLOv8x, I would need to either use its less-mature tracker integration or manage a separate tracking library.

2. **C2PSA Spatial Attention for Retail**: The C2PSA (Cross Stage Partial with Spatial Attention) module in YOLO11 is specifically effective for partially occluded objects — a common scenario in retail aisles where customers are partially hidden behind shelves. This architectural advantage directly addresses our use case.

3. **22% Fewer Parameters**: YOLO11l achieves comparable mAP with significantly fewer parameters. Since we're processing 5 video files, faster per-frame inference translates to meaningful total time savings. This matters for a hackathon where pipeline execution speed affects iteration cycles.

I rejected RT-DETR despite its slightly higher mAP because it lacks the native tracking integration that YOLO11 provides, which would add ~4 hours of tracker integration work.

---

## Decision 2: Event Schema Design — Session-Based with Idempotent Ingestion

### Options Evaluated
| Approach | Pros | Cons |
|----------|------|------|
| Raw frame-level events | Highest granularity | Massive data volume, complex aggregation |
| **Session-based events with types** | **Balanced granularity, clear semantics** | **Requires session logic** |
| Pre-aggregated metrics only | Simple API | Loses raw event traceability |

### What AI Suggested
AI recommended a simple append-only event log with post-hoc aggregation. This would store every frame as an event and compute sessions lazily.

### What I Chose and Why
I designed a **typed event catalogue** (ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL, BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON, REENTRY) with **session_seq** numbering and **idempotent ingestion**.

Key design decisions:

1. **session_seq**: Each event within a visitor's journey gets a sequence number. This enables chronological replay of a visitor's path without relying on timestamp ordering (which can have clock drift issues across cameras). If I need to debug "why did funnel count this visitor twice?", I can trace their event sequence.

2. **confidence is never suppressed**: Every event carries the detection confidence from YOLO11. Low-confidence detections (0.25-0.5) are included rather than filtered because:
   - Filtering creates silent data loss — you can't analyze what you didn't record
   - The API consumer can filter by confidence if needed
   - For anomaly detection, even low-confidence detections carry signal

3. **Idempotent by event_id**: Using `INSERT ... ON CONFLICT (event_id) DO NOTHING` means:
   - Pipeline can be re-run safely without duplicate events
   - Network retries are safe
   - Testing is simpler (reset DB, re-ingest, compare)

4. **metadata is extensible**: The `metadata` JSON field carries event-specific data (queue_depth for BILLING_QUEUE_JOIN, sku_zone for future product-level tracking) without schema migration. This is a deliberate trade-off: structured fields for queried data, flexible JSON for context.

---

## Decision 3: PostgreSQL over SQLite for Event Storage

### Options Evaluated
| Database | Complexity | Concurrent Writes | Query Power | Docker |
|----------|-----------|-------------------|-------------|--------|
| **PostgreSQL** | **Medium** | **Excellent** | **Full SQL + JSONB** | **Official image** |
| SQLite | Very Low | Poor (WAL helps) | Basic SQL | No server needed |
| TimescaleDB | High | Excellent | Time-series optimized | Extension install |
| ClickHouse | High | Excellent | Blazing analytics | Heavy image |

### What AI Suggested
AI initially suggested SQLite "for simplicity" since this is a hackathon project with limited data volume (~50K events at most). The argument: fewer moving parts, no database server to manage, single-file storage.

### What I Chose and Why
I chose **PostgreSQL** despite the higher complexity because:

1. **Concurrent writes matter**: The pipeline writes events while the API serves queries. SQLite's write locking (even with WAL mode) creates contention. PostgreSQL handles this natively with MVCC.

2. **Production-realistic architecture**: The problem statement evaluates "how you'd scale to 40 stores." Submitting with SQLite signals that I haven't considered production deployment. PostgreSQL with asyncpg demonstrates I understand async database patterns used in real FastAPI deployments.

3. **ON CONFLICT clause**: PostgreSQL's `INSERT ... ON CONFLICT DO NOTHING` is the cleanest way to implement idempotent ingestion. SQLite supports this too, but the asyncpg driver ecosystem for FastAPI is more mature than aiosqlite.

4. **Docker Compose alignment**: PostgreSQL has a first-class Docker image with built-in healthchecks (`pg_isready`). The docker-compose.yml becomes cleaner and more standard.

I rejected TimescaleDB because our data volume (~50K events) doesn't justify the added complexity of hypertables and continuous aggregates. I would recommend TimescaleDB if scaling to 40+ stores with millions of daily events — and I document this in DESIGN.md as a scaling consideration.

I rejected ClickHouse because it's designed for analytical workloads at massive scale (billions of rows), and the operational overhead of running it for a hackathon would be counterproductive.

The key insight: **choose the simplest technology that doesn't create a scaling dead-end**. PostgreSQL is that sweet spot between SQLite (too simple) and TimescaleDB/ClickHouse (too complex for current scale).
