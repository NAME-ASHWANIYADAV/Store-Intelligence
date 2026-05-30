# 🛍️ Store Intelligence System

**AI-powered retail analytics platform** that transforms raw CCTV footage into actionable business intelligence: visitor tracking, conversion funnels, zone heatmaps, and real-time anomaly detection.

> **North Star Metric**: Offline Store Conversion Rate  
> `Conversion Rate = Visitors at billing ÷ Total unique floor visitors`

## 🏗️ Architecture

```
CCTV Clips → YOLO11-L + BoT-SORT → ResNet18 ReID → TrackletMerger → Events → FastAPI → Dashboard
```

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Detection | YOLO11-L (CUDA) | Person detection, conf ≥ 0.25 |
| Tracking | BoT-SORT (native ReID) | Multi-object tracking with appearance features |
| Deduplication | ResNet18 512-dim + Union-Find | Track fragment merging (52.5% reduction) |
| Spatial Filter | Binary mask (foot-point) | Excludes non-store areas (glass door reflections) |
| Staff Classifier | Zone-dwell heuristic | Staff if >40% time in staff_zone polygon |
| API | FastAPI + asyncpg + PostgreSQL | Event ingestion, metrics, funnel, anomalies |
| Cache | Redis 7 | Query caching (10s TTL) |
| Dashboard | Streamlit + Plotly | Real-time KPIs, funnel, heatmap, anomalies |
| Deployment | Docker Compose | One-command startup: `docker compose up` |

## 🚀 Quick Start (5 Commands)

### Prerequisites
- Docker & Docker Compose (for API + DB)
- Python 3.11+ with CUDA GPU (for detection pipeline)
- ~4GB GPU VRAM (GTX 1650 or better)

### 1. Clone & Configure

```bash
git clone <repository-url>
cd store-intelligence
cp .env.example .env
```

### 2. Start API + Database + Dashboard

```bash
docker compose up -d
```

This starts:
- **API**: http://localhost:8000 (Swagger docs at `/docs`)
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **Dashboard**: http://localhost:8501

### 3. Run Detection Pipeline (GPU Required)

```bash
# Create virtual environment and install dependencies
python -m venv venv
.\venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt

# Process all CCTV clips → generates events JSONL
python scripts/run_pipeline_v3.py
```

Pipeline output:
- **Events**: `output/v3/STORE_BLR_001_v3.jsonl` (structured event stream)
- **Overlay videos**: `output/v3/CAM_*_final.mp4` (annotated videos showing tracker decisions)

### 4. Ingest Events into API

```bash
python scripts/ingest_events.py
```

### 5. View Dashboard

Open http://localhost:8501

The dashboard auto-refreshes every 5 seconds and shows:
- ⭐ North Star conversion rate (hero metric)
- 📊 6 KPI cards (visitors, entries, floor, engaged, billing, events)
- 🔄 Conversion funnel with drop-off percentages
- 🔥 Zone engagement heatmap (avg dwell time per zone)
- 📈 Event timeline (30s bins, per-camera breakdown)
- 🎯 Event type distribution (donut chart)
- ⏱️ Dwell time histogram
- 🚨 Anomaly detection (queue spike, conversion drop, dead zone)
- 🧬 Track deduplication evolution (v1.0 → v2.1 → v3.0)
- 🗺️ Visitor session detail table

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/events/ingest` | Ingest event batch (up to 500, idempotent by event_id) |
| GET | `/stores/{id}/metrics` | Unique visitors, conversion rate, avg dwell, queue depth |
| GET | `/stores/{id}/funnel` | Entry → Zone Visit → Billing → Purchase (with drop-off %) |
| GET | `/stores/{id}/heatmap` | Zone visit frequency + avg dwell, normalized 0–100 |
| GET | `/stores/{id}/anomalies` | Queue spike, conversion drop, dead zone alerts |
| GET | `/health` | Service status, last event per store, STALE_FEED detection |

### Example

```bash
# Health check
curl http://localhost:8000/health

# Store metrics
curl http://localhost:8000/stores/STORE_BLR_001/metrics

# Conversion funnel
curl http://localhost:8000/stores/STORE_BLR_001/funnel

# Ingest events
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d @output/v3/STORE_BLR_001_v3.jsonl
```

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio aiosqlite httpx

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=html --cov-report=term-missing
```

## 📂 Project Structure

```
store-intelligence/
├── app/                        # FastAPI application
│   ├── main.py                # App entrypoint + lifespan
│   ├── config.py              # Settings (pydantic-settings)
│   ├── database.py            # Async PostgreSQL (asyncpg)
│   ├── models.py              # SQLAlchemy ORM models
│   ├── schemas.py             # Pydantic request/response schemas
│   ├── middleware.py          # Structured logging (structlog)
│   ├── routers/               # API endpoints
│   │   ├── ingest.py          # POST /events/ingest
│   │   ├── metrics.py         # GET /stores/{id}/metrics
│   │   ├── funnel.py          # GET /stores/{id}/funnel
│   │   ├── heatmap.py         # GET /stores/{id}/heatmap
│   │   ├── anomalies.py       # GET /stores/{id}/anomalies
│   │   └── health.py          # GET /health
│   └── services/              # Business logic engines
│       └── anomaly_engine.py  # Anomaly detection rules
├── scripts/                    # Pipeline scripts
│   ├── run_pipeline_v3.py     # Main v3.0 detection pipeline
│   └── ingest_events.py       # Event ingestion into API
├── tracking/                   # ReID + deduplication
│   ├── reid_extractor.py      # ResNet18 embedding extractor
│   └── tracklet_merger.py     # Union-Find + Hungarian merger
├── config/                     # Configuration loaders
│   └── models.py              # Pydantic config models
├── configs/                    # Store YAML configs
│   └── store_001.yaml         # Zone polygons, ignore regions, tripwires
├── dashboard/                  # Streamlit dashboard
│   ├── app.py                 # Dashboard v2.0 (real-time)
│   └── .streamlit/config.toml # Purplle brand theme
├── tests/                      # Test suite
│   ├── test_analytics.py
│   └── ...
├── docs/                       # Documentation
│   ├── DESIGN.md              # Architecture + AI-assisted decisions
│   └── CHOICES.md             # 3 key engineering decisions
├── output/v3/                  # Pipeline outputs
│   ├── STORE_BLR_001_v3.jsonl # Event stream
│   └── CAM_*_final.mp4        # Annotated overlay videos
├── docker-compose.yml          # Full stack (API + PG + Redis + Dashboard)
├── Dockerfile                  # API container
├── requirements.txt            # Python dependencies
└── botsort.yaml               # Tracker configuration
```

## 🔑 Key Design Decisions

| Decision | Chose | Over | Why |
|----------|-------|------|-----|
| Detection | YOLO11-L | YOLOv8x | 22% fewer params, native BoT-SORT ReID |
| ReID | ResNet18 512-dim | Color histograms | 52.5% vs 3.8% dedup rate |
| Track Merger | Union-Find (seq+concurrent) | Sequential only | Handles clustering duplication |
| Staff Detection | Zone-dwell heuristic | Custom CNN | Zero training, config-driven |
| Database | PostgreSQL | SQLite | Concurrent writes, production-ready |
| Dashboard | Streamlit + Plotly | React | 5 hours saved → invested in pipeline |

See [DESIGN.md](docs/DESIGN.md) and [CHOICES.md](docs/CHOICES.md) for detailed reasoning.

## 🎬 Pipeline Performance (v3.0)

```
Camera       Role     Raw Tracks  Merged  Staff  Customers  Events
─────────────────────────────────────────────────────────────────────
CAM_01       floor    19          7       1      6          33
CAM_02       floor    27          8       5      3          160
CAM_03       entry    6           5       0      5          0
CAM_05       billing  9           9       1      8          6
─────────────────────────────────────────────────────────────────────
TOTAL                 61          29      7      22         199

Dedup: 61 → 29 (52.5% reduction)
Total processing time: 2,222s (~37 minutes on GTX 1650)
```

## 📄 License

Built for Purplle Tech Challenge 2026. Challenge use only.
