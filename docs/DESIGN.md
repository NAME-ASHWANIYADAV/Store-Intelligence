# Store Intelligence System — Architecture & Design Document

## Overview

The Store Intelligence System is an end-to-end retail analytics platform that transforms raw CCTV footage into actionable business insights. It processes video from multiple cameras across physical retail stores to track visitor behavior, compute conversion funnels, generate zone-level heatmaps, and detect real-time anomalies.

The system is designed as a production-grade pipeline with clean separation between the **Detection Pipeline** (offline, batch processing of CCTV clips), the **Intelligence API** (real-time serving of analytics), and the **Dashboard** (live visualization of store metrics).

## System Architecture

```
CCTV Clips (4 cameras, ~680MB total)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│              DETECTION PIPELINE v3.0                       │
│                                                            │
│  YOLO11-L ──► BoT-SORT ──► SpatialFilter ──► ReIDExtract │
│  (Person     (Multi-Obj    (Ignore mask     (ResNet18     │
│   Detection)  Tracking)     foot-point)      512-dim)     │
│                    │                             │         │
│                    ▼                             ▼         │
│             TrackletRecord ◄──── Embeddings (L2-norm)     │
│                    │                                       │
│                    ▼                                       │
│  TrackletMerger (Union-Find + Hungarian Matching)          │
│    Mode 1: Sequential (gap-based re-association)           │
│    Mode 2: Concurrent (clustering dedup)                   │
│    Cost: 55% appearance + 20% spatial + 15% temporal       │
│           + 10% velocity                                   │
│                    │                                       │
│                    ▼                                       │
│  StaffClassifier ──► ZoneEngine ──► LineCrossing           │
│  (zone dwell       (enter/exit     (entry/exit             │
│   heuristic)        with dwell)     FSM)                   │
│                    │                                       │
│                    ▼                                       │
│              JSONL Events ──► POST /events/ingest          │
└───────────────────────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────────┐
│           INTELLIGENCE API (FastAPI + asyncpg)             │
│                                                            │
│  POST /events/ingest    ──► PostgreSQL (idempotent)        │
│  GET  /stores/{id}/metrics  ──► Real-time aggregation      │
│  GET  /stores/{id}/funnel   ──► Session-based conversion   │
│  GET  /stores/{id}/heatmap  ──► Zone visit frequency       │
│  GET  /stores/{id}/anomalies ──► Rule-based detection      │
│  GET  /health               ──► Feed freshness + status    │
│                                                            │
│  Middleware: structlog JSON logging (trace_id, latency_ms) │
│  Error handling: no raw stack traces in production         │
└───────────────────┬───────────────────────────────────────┘
                    │  HTTP polling (5s interval)
                    ▼
┌───────────────────────────────────────────────────────────┐
│           STREAMLIT DASHBOARD v2.0                         │
│                                                            │
│  ⭐ North Star: Conversion Rate (hero metric)              │
│  📊 6 KPI Cards (visitors, entries, engaged, billing)      │
│  🔄 Conversion Funnel with drop-off percentages            │
│  🔥 Zone Engagement Heatmap (avg dwell per zone)           │
│  📈 Event Timeline (30s bins, per-camera)                  │
│  🎯 Event Type Distribution (donut chart)                  │
│  ⏱️  Dwell Time Histogram (per-zone distribution)           │
│  🚨 Anomaly Panel (queue spike, conversion drop, dead zone)│
│  🧬 Dedup Evolution (v1→v2.1→v3.0 comparison)             │
│  🗺️  Visitor Session Detail Table                           │
│  ⚡ Real-time auto-refresh (5s with pulsing indicator)      │
└───────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Video Ingestion**: CCTV clips are processed offline by the detection pipeline
2. **Detection**: YOLO11-L detects all persons in each frame (class=0, vid_stride=2)
3. **Tracking**: BoT-SORT maintains consistent track IDs with native ReID enabled
4. **Spatial Filtering**: Binary mask using foot-point (bottom-center of bbox) filters out non-store areas (e.g., street pedestrians through glass door on CAM_03)
5. **ReID Embedding**: ResNet18 (pretrained, headless) extracts 512-dim L2-normalized features every 3rd detection frame. Batch inference on GPU for efficiency (~2ms per crop on GTX 1650)
6. **Track Deduplication**: TrackletMerger uses Union-Find with two modes:
   - **Sequential**: Merges fragments where person disappears behind shelves and reappears (gap-based, up to 12 seconds)
   - **Concurrent**: Merges duplicate IDs created when people cluster together (spatial overlap, short fragment absorbed into long track)
   - **Cost function**: α·appearance(0.55) + β·temporal(0.15) + γ·spatial(0.20) + δ·velocity(0.10)
   - **Result**: 52.5% track fragmentation reduction (61 raw → 29 real persons)
7. **Staff Classification**: Zone-time heuristic — person in `staff_zone` polygon for >40-45% of their total tracking duration → classified as staff
8. **Zone Events**: Enter/exit/dwell events emitted per zone polygon with configurable dwell thresholds
9. **Line Crossing**: Finite State Machine for entry/exit at doorway tripwire with cooldown and min-pixel-distance
10. **Event Emission**: All events written to JSONL file, then POST'd to /events/ingest
11. **Session Building**: Events aggregated into visitor sessions on ingest
12. **Analytics**: Metrics, funnel, heatmap, anomalies computed on-demand
13. **Dashboard**: Streamlit reads JSONL directly (offline mode) or polls API (live mode)

## AI-Assisted Design Decisions

### 1. Detection Model Selection — YOLO11-L over YOLOv8x

I used AI (Claude, Gemini, Kimi, DeepSeek) to evaluate detection model options. AI initially recommended YOLOv8x as the "battle-tested gold standard." I chose YOLO11-L instead after independent research showed:

- **22% fewer parameters** (25.3M vs 68.2M) with comparable mAP on COCO
- **Native BoT-SORT + ReID integration** — no separate tracker package needed. Just `model.track(tracker="botsort.yaml", persist=True)` with `with_reid: True`
- **C2PSA spatial attention** — specifically helps with partial occlusion, common in retail aisles
- **Architecture refinement**: C3k2 modules replace C2f blocks for richer spatial features

AI agreed after I presented these benchmarks. The decision saved significant development time.

### 2. Track Deduplication — ResNet18 + Union-Find over Color Histograms

My initial implementation used **192-dim color histograms** (HSV, 64 bins per channel) for appearance matching between fragmented tracklets. AI (Claude) suggested this as "simple and effective."

I rejected it after empirical testing showed only **3.8% dedup rate** — nearly useless. The reason: color histograms are invariant to spatial structure. Two different people wearing similar-colored clothes produce near-identical histograms.

I replaced it with **ResNet18** (pretrained on ImageNet, final FC layer removed, 512-dim output, L2-normalized). The results:

| Approach | Dedup Rate | CAM_01 Tracks | CAM_02 Tracks |
|----------|-----------|---------------|---------------|
| No dedup (v1.0) | 0% | 16 | 21 |
| Color histogram (v2.0) | 3.8% | 15 | 20 |
| ResNet18 sequential (v2.1) | 41.0% | 10 | 12 |
| **ResNet18 + concurrent (v3.0)** | **52.5%** | **7** | **8** |

The key insight: histograms encode *what colors exist* but not *where they are*. CNNs encode spatial structure (face position, clothing patterns, body proportions). AI helped me evaluate the CNN options (MobileNetV2 vs ResNet18 vs EfficientNet-B0), and I chose ResNet18 for its balance of accuracy and inference speed on a GTX 1650.

### 3. Concurrent Track Merger — Novel Contribution

AI tools (Claude, DeepSeek, Gemini) all suggested **sequential-only** merging: track B starts after track A ends → check if they're the same person. None suggested handling **concurrent overlapping tracks**.

I discovered through video analysis that in billing counter areas (CAM_05), the tracker assigns **duplicate IDs when people stand close together**. Person A gets IDs #200 and #203 simultaneously because the bounding boxes overlap during clustering.

I designed **Mode 2 (Concurrent Merge)**: if two tracks overlap in time, one is significantly shorter than the other (< 50% duration), they're spatially within 150px, and appearance similarity > 0.3 cosine — merge the shorter into the longer.

This is an example of where I **disagreed with all 4 AI tools** and designed a novel solution based on domain-specific observation. It reduced CAM_01 from 10 → 7 tracks and CAM_02 from 12 → 8 tracks.

### 4. Staff Detection — Dwell Heuristic over Custom CNN

AI suggested training a custom CNN classifier for uniform detection. I rejected this because:

- **No labeled training data** available in the challenge dataset
- **Dwell-time heuristic handles 80% of cases**: staff stays in `staff_zone` polygon for >40-45% of their tracking duration
- **Zone polygons are configurable per-camera** in YAML — no retraining needed for new stores
- **Zero training required**: deployable immediately

### 5. Dashboard — Streamlit over React

AI recommended React + WebSocket for "maximum wow factor." I overrode this:

- Dashboard is worth +10 bonus points; Detection pipeline is worth 30 points
- React would cost ~8 hours vs Streamlit ~3 hours
- The 5 hours saved were invested in the ResNet18 ReID pipeline upgrade
- Streamlit with plotly + custom CSS still produces a professional dashboard with real-time auto-refresh

## Configuration-Driven Architecture

A major design principle: **all spatial parameters (zones, ignore regions, tripwires, staff zones) are stored in a single YAML file** (`configs/store_001.yaml`) using **normalized coordinates (0.0-1.0)**.

Benefits:
- **Resolution independence**: Same config works at 720p, 1080p, or 4K
- **No code changes for new stores**: Just create `store_002.yaml`
- **Version controllable**: YAML diffs in git show exactly what spatial regions changed
- **Edge-ready**: config file ships with the edge device, no cloud dependency

## Production Considerations

### Observability
- Every API request logged with: `trace_id`, `store_id`, `endpoint`, `latency_ms`, `status_code`
- structlog produces machine-parseable JSON logs
- Health endpoint reports per-store feed freshness with STALE_FEED detection (>10 min lag)

### Graceful Degradation
- Database unavailable → HTTP 503 with structured error body (no stack traces)
- Redis unavailable → fallback to direct database queries (slower but functional)
- No raw exception details in production responses

### Idempotency
- Event ingestion uses `INSERT ... ON CONFLICT (event_id) DO NOTHING`
- Safe to replay events without data corruption
- Enables retry-safe pipeline execution

## Scaling Considerations (40-Store Projection)

At 40 live stores (120 cameras), the first bottleneck would be **PostgreSQL connection pool exhaustion**. Mitigations:

1. **Connection pooling**: asyncpg with pool_size=20, max_overflow=10
2. **Redis caching**: Metrics computed on ingest, cached with 10s TTL
3. **Session pre-computation**: Sessions rebuilt on ingest (not on query) — O(1) reads
4. **Event partitioning**: Table partitioned by store_id + month for query locality
5. **Edge inference**: Detection pipeline runs on edge devices (GTX 1650 class), only events stream to cloud. ~2KB per event vs ~100MB per minute of raw video = 50,000x bandwidth reduction.
