# Store Intelligence System — Technical Choices

This document captures the three most impactful technical decisions made during development, including what AI tools suggested, what I ultimately chose, and the reasoning behind each override or acceptance.

---

## Decision 1: Detection Model — YOLO11-L with Native BoT-SORT

### Options Evaluated
| Model | mAP (COCO) | Parameters | Speed | Key Feature |
|-------|-----------|------------|-------|-------------|
| YOLOv8x | ~53.9 | 68.2M | Baseline | Battle-tested, largest community |
| YOLOv8l | ~52.9 | 43.7M | Faster | Smaller but lower accuracy |
| **YOLO11-L** | **~53.4** | **~25.3M** | **Faster** | **Native ReID, C2PSA attention** |
| YOLOv12 | Highest | Very high | Slow | Research-grade, unstable |
| RT-DETR | ~54.3 | 42M | Moderate | Transformer-based, NMS-free |

### What AI Suggested
Claude initially recommended YOLOv8x as the "production standard" with the largest community. Gemini suggested considering RT-DETR for its transformer architecture.

### What I Chose and Why
I chose **YOLO11-L** for three specific reasons:

1. **Native BoT-SORT with ReID**: YOLO11 includes BoT-SORT as a built-in tracker with `with_reid: True` support. This eliminates an entire dependency (separate BoT-SORT or Deep OC-SORT installation), reduces integration bugs, and simplifies the codebase. With YOLOv8x, I would need to manage a separate tracking library.

2. **C2PSA Spatial Attention for Retail**: The C2PSA (Cross Stage Partial with Spatial Attention) module in YOLO11 is specifically effective for partially occluded objects — a common scenario in retail aisles where customers are partially hidden behind shelves and displays. This architectural advantage directly addresses our use case.

3. **22% Fewer Parameters**: YOLO11-L achieves comparable mAP with significantly fewer parameters. On a GTX 1650 (4GB VRAM), this translates to faster inference and lower memory pressure, leaving room for the ResNet18 ReID model to run concurrently on the same GPU.

I rejected RT-DETR despite its slightly higher mAP because it lacks the native tracking integration that YOLO11 provides, which would add ~4 hours of tracker integration work.

### Validation
- Processed 4 cameras (15,868 total frames) in ~1,100 seconds of detection time
- Detected 61 raw person tracks across the store with high recall
- No false positives from non-person objects (shelves, mannequins, displays)

---

## Decision 2: ReID Architecture — ResNet18 over Color Histograms

### Options Evaluated
| Approach | Dimensions | Accuracy | Speed | Training Required |
|----------|-----------|----------|-------|-------------------|
| Color histogram (HSV) | 192 | Low | Fast | None |
| MobileNetV2 | 1280 | Medium | Fast | None (pretrained) |
| **ResNet18** | **512** | **High** | **~2ms/crop** | **None (pretrained)** |
| EfficientNet-B0 | 1280 | Higher | Slower | None (pretrained) |
| OSNet (torchreid) | 512 | Highest | Moderate | Person ReID pretrained |

### What AI Suggested
AI initially recommended simple color histogram matching (HSV, 64 bins per channel) for "fast and lightweight ReID without any model dependencies." Multiple AI tools (Claude, Gemini) endorsed this approach.

### What I Chose and Why

I started with color histograms and it was a **disaster**: only **3.8% dedup rate** across the store. The fundamental problem is that histograms encode *what colors exist* but not *where they are*. Two different people wearing similar-colored clothing produce near-identical 192-dim vectors.

I replaced it with **ResNet18** (pretrained on ImageNet, final FC layer removed, 512-dim L2-normalized output) after benchmarking:

| Version | Approach | Total Tracks | Dedup Rate |
|---------|----------|-------------|-----------|
| v1.0 | No dedup | 83 | 0% |
| v2.0 | Color histogram | 80 | 3.8% |
| v2.1 | ResNet18 sequential | 36 | 41.0% |
| **v3.0** | **ResNet18 + concurrent** | **29** | **52.5%** |

**Why ResNet18 specifically** (not MobileNetV2 or OSNet):
- **512 dimensions** is a sweet spot — enough for discriminative power, small enough for fast cosine distance computation in the merger's O(n²) loop
- **~2ms per crop on GTX 1650** — fits within the frame budget even at vid_stride=2
- **ImageNet pretrained weights** are surprisingly effective for person appearance because they capture clothing textures, body proportions, and spatial patterns
- **No person-ReID specific training needed** — OSNet would be slightly better but requires torchreid dependency and person-ReID weights, adding complexity

### Key Insight (Where I Disagreed with AI)
All 4 AI tools suggested **sequential-only** track merging. I discovered through manual video analysis that the billing counter camera (CAM_05) creates **concurrent duplicate IDs** when people cluster together. I designed a novel "Mode 2: Concurrent Merge" that handles overlapping tracks — this reduced CAM_01 from 10→7 and CAM_02 from 12→8 tracks.

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

1. **Concurrent writes matter**: The pipeline writes events while the API serves queries. SQLite's write locking (even with WAL mode) creates contention that manifests as "database is locked" errors under load. PostgreSQL handles this natively with MVCC — multiple writers and readers operate without blocking each other.

2. **Production-realistic architecture**: The problem statement evaluates "how you'd scale to 40 stores." Submitting with SQLite signals that I haven't considered production deployment. PostgreSQL with asyncpg demonstrates I understand async database patterns used in real FastAPI deployments.

3. **ON CONFLICT clause**: PostgreSQL's `INSERT ... ON CONFLICT DO NOTHING` is the cleanest way to implement idempotent ingestion. Combined with asyncpg's native async support, this gives us non-blocking, replay-safe event ingestion.

4. **Docker Compose alignment**: PostgreSQL has a first-class Docker image with built-in healthchecks (`pg_isready`). The docker-compose.yml becomes cleaner with proper service dependency chains: postgres (healthy) → api → dashboard.

5. **JSONB for metadata**: The event schema includes a flexible `metadata` field (queue_depth, sku_zone, session_seq). PostgreSQL's JSONB type allows us to query inside this field efficiently, while SQLite would require text parsing.

### What I Would Change at Scale
I would recommend **TimescaleDB** (PostgreSQL extension) if scaling to 40+ stores with millions of daily events. Hypertables with time-based partitioning and continuous aggregates would make the metrics and funnel queries 10-100x faster. I didn't use it here because the operational overhead is unjustified for ~200 events across 4 cameras.

I rejected ClickHouse because it's designed for analytical workloads at massive scale (billions of rows), and the operational overhead of running it for a hackathon would be counterproductive.

**The key insight: choose the simplest technology that doesn't create a scaling dead-end.** PostgreSQL is that sweet spot between SQLite (too simple, locks under concurrent writes) and TimescaleDB/ClickHouse (too complex for current scale).
