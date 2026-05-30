# 🏪 Store Intelligence System

Real-time retail analytics platform powered by CCTV-based computer vision. Transforms raw store footage into actionable business insights: visitor tracking, conversion funnels, zone heatmaps, and anomaly detection.

## 🏗️ Architecture

```
CCTV Clips → YOLO11l + BoT-SORT → Event Stream → FastAPI → PostgreSQL → Dashboard
```

- **Detection**: YOLO11l with native BoT-SORT tracking + ReID
- **API**: FastAPI with async PostgreSQL (asyncpg) + Redis caching
- **Dashboard**: Streamlit with real-time plotly charts
- **Infrastructure**: Docker Compose (API + PostgreSQL + Redis + Dashboard)

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for pipeline only)
- GPU recommended (for YOLO11l detection)

### 1. Clone & Configure

```bash
git clone <repository-url>
cd store-intelligence
cp .env.example .env
# Edit .env with your API keys if needed
```

### 2. Start Services

```bash
docker compose up -d
```

This starts:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

### 3. Run Detection Pipeline

```bash
# Install pipeline dependencies (run once)
pip install -r requirements.txt

# Process all CCTV clips → generate events
python -m pipeline.run

# Ingest events into API
python scripts/ingest_events.py
```

### 4. View Dashboard

Open http://localhost:8501 to see the live dashboard with:
- KPI cards (visitors, conversion, dwell, queue depth)
- Conversion funnel chart
- Zone heatmap
- Anomaly alerts

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/events/ingest` | Ingest event batch (up to 500, idempotent) |
| GET | `/stores/{id}/metrics` | Real-time store metrics |
| GET | `/stores/{id}/funnel` | Conversion funnel |
| GET | `/stores/{id}/heatmap` | Zone visit heatmap |
| GET | `/stores/{id}/anomalies` | Anomaly alerts |
| GET | `/health` | System health check |

## 🧪 Running Tests

```bash
# Install test dependencies
pip install aiosqlite

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## 📂 Project Structure

```
store-intelligence/
├── app/                    # FastAPI application
│   ├── main.py            # App entrypoint
│   ├── config.py          # Settings (pydantic-settings)
│   ├── database.py        # Async PostgreSQL
│   ├── models.py          # SQLAlchemy ORM models
│   ├── schemas.py         # Pydantic schemas
│   ├── middleware.py       # Structured logging
│   ├── routers/           # API endpoints
│   └── services/          # Business logic engines
├── pipeline/              # Detection pipeline
│   ├── run.py            # Orchestrator
│   ├── detect.py         # YOLO11l + BoT-SORT
│   ├── zones.py          # Zone detection
│   ├── direction.py      # Entry/exit direction
│   ├── staff_detector.py # Staff classification
│   └── event_emitter.py  # JSONL event generation
├── dashboard/             # Streamlit dashboard
│   └── app.py
├── tests/                 # Comprehensive test suite
├── docs/                  # Architecture docs
│   ├── DESIGN.md         # System design + AI decisions
│   └── CHOICES.md        # Technical choices
├── docker-compose.yml     # Full stack deployment
├── Dockerfile
└── requirements.txt
```

## 🔑 Key Design Decisions

1. **YOLO11l over YOLOv8x**: 22% fewer parameters, native BoT-SORT + ReID, C2PSA attention
2. **Dwell + VLM staff detection**: No labeled training data needed
3. **Streamlit over React**: 5 hours saved, invested in detection accuracy
4. **PostgreSQL over SQLite**: Concurrent writes, production-realistic

See [DESIGN.md](docs/DESIGN.md) and [CHOICES.md](docs/CHOICES.md) for detailed reasoning.

## 📄 License

MIT License. Built for Purplle Tech Challenge 2026.
