"""
Store Intelligence System - Streamlit Live Dashboard
Real-time store analytics with KPIs, funnel, heatmap, and anomaly feed.
"""

import os
import time
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ===== Configuration =====
API_URL = os.getenv("API_URL", "http://localhost:8000")
REFRESH_INTERVAL = 5  # seconds

# ===== Page Config =====
st.set_page_config(
    page_title="Store Intelligence Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== Custom CSS for Premium Dark Theme =====
st.markdown("""
<style>
    /* Dark premium theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* KPI Card styling */
    .kpi-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d4ff;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Anomaly cards */
    .anomaly-critical {
        background: rgba(255,59,48,0.15);
        border-left: 4px solid #ff3b30;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .anomaly-warn {
        background: rgba(255,204,0,0.15);
        border-left: 4px solid #ffcc00;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .anomaly-info {
        background: rgba(0,122,255,0.15);
        border-left: 4px solid #007aff;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }

    /* Header */
    .dashboard-header {
        text-align: center;
        padding: 20px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 24px;
    }
    .dashboard-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def fetch_api(endpoint: str):
    """Fetch data from API with error handling."""
    try:
        resp = requests.get(f"{API_URL}{endpoint}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def render_kpi_card(label: str, value: str, icon: str):
    """Render a styled KPI card."""
    st.markdown(f"""
    <div class="kpi-card">
        <div style="font-size: 1.5rem; margin-bottom: 4px;">{icon}</div>
        <p class="kpi-value">{value}</p>
        <p class="kpi-label">{label}</p>
    </div>
    """, unsafe_allow_html=True)


def render_funnel(funnel_data: dict):
    """Render conversion funnel as a plotly funnel chart."""
    if not funnel_data or not funnel_data.get("stages"):
        st.info("No funnel data available yet.")
        return

    stages = funnel_data["stages"]
    fig = go.Figure(go.Funnel(
        y=[s["name"] for s in stages],
        x=[s["count"] for s in stages],
        textinfo="value+percent initial",
        textposition="inside",
        marker=dict(
            color=["#00d4ff", "#7b2ff7", "#ff6b6b", "#ffd93d"],
        ),
        connector=dict(line=dict(color="rgba(255,255,255,0.2)", width=1)),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=14),
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
        title=dict(text="Conversion Funnel", font=dict(size=18)),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_heatmap(heatmap_data: dict):
    """Render zone heatmap as a horizontal bar chart."""
    if not heatmap_data or not heatmap_data.get("zones"):
        st.info("No heatmap data available yet.")
        return

    zones = heatmap_data["zones"]
    df = pd.DataFrame(zones)

    fig = px.bar(
        df, x="intensity", y="zone_id",
        orientation="h",
        color="intensity",
        color_continuous_scale=["#1a1a2e", "#7b2ff7", "#ff6b6b", "#ffd93d"],
        labels={"intensity": "Intensity (0-100)", "zone_id": "Zone"},
        text="visit_count",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=13),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        title=dict(text="Zone Heatmap", font=dict(size=18)),
        showlegend=False,
        xaxis=dict(range=[0, 105]),
    )
    fig.update_traces(textposition="outside", textfont_size=12)
    st.plotly_chart(fig, use_container_width=True)

    # Data confidence badge
    confidence = heatmap_data.get("data_confidence", "LOW")
    color = "#4caf50" if confidence == "HIGH" else "#ff9800"
    st.markdown(
        f'<span style="background:{color};color:white;padding:4px 12px;'
        f'border-radius:12px;font-size:0.8rem;">Data Confidence: {confidence}</span>',
        unsafe_allow_html=True,
    )


def render_anomalies(anomaly_data: dict):
    """Render anomaly alerts as styled cards."""
    if not anomaly_data or not anomaly_data.get("anomalies"):
        st.success("✅ No anomalies detected — all systems normal.")
        return

    for anomaly in anomaly_data["anomalies"]:
        severity = anomaly.get("severity", "INFO")
        css_class = f"anomaly-{severity.lower()}"
        icon = {"CRITICAL": "🔴", "WARN": "🟡", "INFO": "🔵"}.get(severity, "⚪")

        st.markdown(f"""
        <div class="{css_class}">
            <strong>{icon} {anomaly['type']}</strong> — <em>{severity}</em><br>
            <span style="color:rgba(255,255,255,0.8);">{anomaly['suggested_action']}</span><br>
            <small style="color:rgba(255,255,255,0.4);">
                Current: {anomaly['current_value']} | Baseline: {anomaly['baseline']} |
                Deviation: {anomaly['deviation_sigma']}σ
            </small>
        </div>
        """, unsafe_allow_html=True)


# ===== Sidebar =====
with st.sidebar:
    st.markdown("### 🏪 Store Intelligence")
    st.markdown("---")

    # Fetch available stores from health endpoint
    health = fetch_api("/health")
    store_ids = []
    if health and health.get("stores"):
        store_ids = list(health["stores"].keys())

    if not store_ids:
        store_ids = ["STORE_BLR_001", "STORE_BLR_002"]

    store_id = st.selectbox("Select Store", store_ids)
    auto_refresh = st.toggle("Auto-Refresh (5s)", value=True)

    st.markdown("---")

    # System Health
    if health:
        db_color = "🟢" if health.get("database") == "connected" else "🔴"
        redis_color = "🟢" if health.get("redis") == "connected" else "🔴"
        st.markdown(f"**System Health**")
        st.markdown(f"{db_color} Database: {health.get('database', 'unknown')}")
        st.markdown(f"{redis_color} Redis: {health.get('redis', 'unknown')}")
        st.markdown(f"⏱️ Uptime: {health.get('uptime_seconds', 0):.0f}s")
    else:
        st.error("⚠️ API unreachable")

    st.markdown("---")
    st.markdown(
        '<small style="color:rgba(255,255,255,0.3);">'
        'Store Intelligence v1.0.0<br>Purplle Tech Challenge 2026</small>',
        unsafe_allow_html=True,
    )


# ===== Main Dashboard =====
st.markdown("""
<div class="dashboard-header">
    <h1 class="dashboard-title">🏪 Store Intelligence Dashboard</h1>
    <p style="color:rgba(255,255,255,0.5);margin-top:4px;">
        Real-time retail analytics powered by YOLO11 + BoT-SORT
    </p>
</div>
""", unsafe_allow_html=True)

# Create placeholder for dynamic content
placeholder = st.empty()

while True:
    with placeholder.container():
        # ===== KPI Row =====
        metrics = fetch_api(f"/stores/{store_id}/metrics")

        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_kpi_card(
                    "Unique Visitors",
                    str(metrics.get("unique_visitors", 0)),
                    "👥"
                )
            with col2:
                rate = metrics.get("conversion_rate", 0)
                render_kpi_card(
                    "Conversion Rate",
                    f"{rate:.1%}",
                    "💰"
                )
            with col3:
                # Average dwell across all zones
                dwells = metrics.get("avg_dwell_per_zone", {})
                avg_dwell = sum(dwells.values()) / max(len(dwells), 1) if dwells else 0
                render_kpi_card(
                    "Avg Dwell Time",
                    f"{avg_dwell/1000:.0f}s",
                    "⏱️"
                )
            with col4:
                render_kpi_card(
                    "Queue Depth",
                    str(metrics.get("current_queue_depth", 0)),
                    "🚶"
                )
        else:
            st.warning("⏳ Waiting for metrics data...")

        st.markdown("<br>", unsafe_allow_html=True)

        # ===== Charts Row =====
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            funnel = fetch_api(f"/stores/{store_id}/funnel")
            render_funnel(funnel)

        with chart_col2:
            heatmap = fetch_api(f"/stores/{store_id}/heatmap")
            render_heatmap(heatmap)

        st.markdown("<br>", unsafe_allow_html=True)

        # ===== Anomaly Feed =====
        st.markdown("### 🚨 Anomaly Alerts")
        anomalies = fetch_api(f"/stores/{store_id}/anomalies")
        render_anomalies(anomalies)

        # ===== Zone Dwell Details =====
        if metrics and metrics.get("avg_dwell_per_zone"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Zone Dwell Details")
            dwell_df = pd.DataFrame([
                {"Zone": k, "Avg Dwell (sec)": round(v / 1000, 1)}
                for k, v in metrics["avg_dwell_per_zone"].items()
            ])
            if not dwell_df.empty:
                st.dataframe(dwell_df, use_container_width=True, hide_index=True)

    # Auto-refresh control
    if not auto_refresh:
        break
    time.sleep(REFRESH_INTERVAL)
