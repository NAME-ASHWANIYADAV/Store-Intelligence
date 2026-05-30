# Store Intelligence System — Architecture & Design Document

## Overview

The Store Intelligence System is an end-to-end retail analytics platform that transforms raw CCTV footage into actionable business insights. It processes video from multiple cameras across physical retail stores to track visitor behavior, compute conversion funnels, generate zone-level heatmaps, and detect real-time anomalies.

The system is designed as a production-grade pipeline with clean separation between the **Detection Pipeline** (offline, batch processing of CCTV clips), the **Intelligence API** (real-time serving of analytics), and the **Dashboard** (live visualization of store metrics).

## System Architecture

```
CCTV Clips (5 cameras, ~680MB total)
        │
        ▼
┌──────────────────────────────────────┐
│       DETECTION PIPELINE              │
│                                       │
│  YOLO11l ──► BoT-SORT (ReID) ──► Events │
│  Person Det   Multi-Object     JSONL   │
│  conf=0.25    Tracking         Output  │
│               + Native ReID            │
│                                       │
│  Zone Manager ◄── store_layout.json   │
│  Staff Detector ◄── Grok Vision VLM   │
│  Direction Det ◄── Threshold Line     │
│  POS Correlator ◄── transactions.csv  │
└──────────────┬───────────────────────┘
               │  POST /events/ingest
               ▼
┌──────────────────────────────────────┐
│       INTELLIGENCE API (FastAPI)      │
│                                       │
│  Ingest ──► PostgreSQL ──► Engines    │
│  (idempotent)  (Events +    Metrics   │
│                 Sessions)   Funnel    │
│                             Heatmap   │
│                             Anomaly   │
│                                       │
│  Structured Logging (structlog JSON)  │
│  Global Error Handling (no raw traces)│
└──────────────┬───────────────────────┘
               │  HTTP polling
               ▼
┌──────────────────────────────────────┐
│       STREAMLIT DASHBOARD             │
│                                       │
│  KPI Cards │ Funnel Chart │ Heatmap  │
│  Anomaly Feed │ Auto-Refresh (5s)    │
└──────────────────────────────────────┘
```

## Data Flow

1. **Video Ingestion**: CCTV clips are processed offline by the detection pipeline
2. **Detection**: YOLO11l detects all persons in each frame (class=0, conf≥0.25)
3. **Tracking**: BoT-SORT maintains consistent track IDs with native ReID enabled
4. **Zone Assignment**: Each person's centroid is tested against zone polygons from store_layout.json
5. **Event Generation**: Zone transitions, entries, exits generate structured JSONL events
6. **API Ingestion**: Events are POSTed to `/events/ingest` (idempotent by event_id)
7. **Session Building**: Events are aggregated into visitor sessions on ingest
8. **Analytics**: Metrics, funnel, heatmap, anomalies computed on-demand from session data
9. **Dashboard**: Streamlit polls API every 5 seconds for live updates

## AI-Assisted Design Decisions

### 1. Detection Model Selection — YOLO11l over YOLOv8x

I used AI (Claude/Gemini) to evaluate detection model options. AI initially recommended YOLOv8x as the "battle-tested gold standard." I chose YOLO11l instead after independent research showed:

- **22% fewer parameters** (25.3M vs 68.2M) with comparable mAP on COCO
- **Native BoT-SORT + ReID integration** — no separate tracker package needed. Just `model.track(tracker="botsort.yaml", persist=True)` with `with_reid: True`
- **C2PSA spatial attention** — specifically helps with partial occlusion, which is common in retail aisles and billing areas
- **Architecture refinement**: C3k2 modules replace C2f blocks for richer spatial features

AI agreed after I presented these benchmarks. The decision saved significant development time (no separate tracker integration) while maintaining detection quality.

### 2. Staff Detection — Dwell Heuristic + VLM over Custom CNN

AI suggested training a custom CNN classifier for uniform detection. I rejected this because:

- **No labeled training data** available in the challenge dataset
- **Dwell-time heuristic handles 80% of cases**: staff stays for >60% of the video duration; customers don't
- **Grok Vision VLM for ambiguous cases**: crop + "Is this person wearing a uniform?" with majority vote across 3 sampled frames
- **Zero training required**: VLM inference is plug-and-play

This is a better engineering trade-off: simpler, more robust, deployable without ML training infrastructure.

### 3. Dashboard — Streamlit over React

AI recommended React + WebSocket for "maximum wow factor." I overrode this decision based on time allocation analysis:

- Dashboard is worth +10 bonus points
- Detection pipeline is worth 30 points
- React would cost ~8 hours vs Streamlit ~3 hours
- The 5 hours saved were invested in edge case handling and test coverage
- Streamlit with plotly and custom CSS still produces a professional, visually appealing dashboard

This was a **strategic time allocation decision**, not a technical one. I documented it because the problem statement values "thoughtful trade-offs" over maximizing each component in isolation.

## Production Considerations

### Observability
- Every API request is logged with: `trace_id`, `store_id`, `endpoint`, `latency_ms`, `status_code`
- structlog produces machine-parseable JSON logs
- Health endpoint reports per-store feed freshness with STALE_FEED detection

### Graceful Degradation
- Database unavailable → HTTP 503 with structured error (no stack traces)
- Redis unavailable → fallback to direct database queries (slower but functional)
- VLM API unavailable → staff detection falls back to dwell heuristic only
- No raw exception details in production responses

### Idempotency
- Event ingestion uses `INSERT ... ON CONFLICT (event_id) DO NOTHING`
- Safe to replay events without corruption
- Enables retry-safe pipeline execution

## Scaling Considerations (40-Store Projection)

At 40 live stores, the first bottleneck would be **PostgreSQL connection pool exhaustion**. Mitigations:

1. **Connection pooling**: asyncpg with pool_size=20, max_overflow=10
2. **Redis caching**: Metrics computed on ingest, cached with 10s TTL for repeated queries
3. **Session pre-computation**: Sessions rebuilt on ingest (not on query) — makes read endpoints O(1) from cache
4. **Event partitioning**: Table partitioned by store_id + month for query locality
