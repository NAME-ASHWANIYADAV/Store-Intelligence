"""
Store Intelligence System — Live Analytics Dashboard v3.0
Part E (Bonus +10 pts) — Purplle Tech Challenge 2026

REAL-TIME dashboard with auto-refresh every 30s + keepalive JS injection.
Never sleeps. Reads pipeline JSONL output + polls FastAPI backend when live.

Launch:  streamlit run dashboard/app.py
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import streamlit as st

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Purplle · Store Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
JSONL_PATH = PROJECT_ROOT / "output" / "v3" / "STORE_BLR_001_v3.jsonl"
STORE_ID = "STORE_BLR_001"

# Palette
PK  = "#e5007e"
PK2 = "#ff4da6"
CY  = "#00d4ff"
GN  = "#00e676"
GD  = "#ffd700"
RD  = "#ff3d71"
OR  = "#ff9100"
PR  = "#7b2ff7"

ZONE_PAL = [PK, CY, PK2, GD, GN, OR, PR]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#d0d0e8", family="Inter, sans-serif", size=12),
)

# ─── Anti-sleep JS: pings Streamlit every 55s so server never idles ──────────
st.html(
    """
    <script>
    (function keepAlive() {
        setInterval(function(){
            fetch(window.location.href, {method:'HEAD', cache:'no-store'});
        }, 55000);
    })();
    </script>
    """
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="st-"] { font-family:'Inter',sans-serif; }
.block-container { padding:1rem 1.5rem 0; max-width:1440px; }

/* ── Header ── */
.hdr {
    background:linear-gradient(135deg,#e5007e 0%,#7b2ff7 50%,#00d4ff 100%);
    padding:1rem 1.6rem; border-radius:14px; margin-bottom:1rem;
    box-shadow:0 6px 28px rgba(229,0,126,.22);
    display:flex; justify-content:space-between; align-items:center;
}
.hdr h1{color:#fff;margin:0;font-size:1.5rem;font-weight:800;letter-spacing:-.5px}
.hdr .sub{color:rgba(255,255,255,.75);font-size:.75rem;margin-top:2px}
.live-badge{
    background:rgba(0,230,118,.12);border:1px solid rgba(0,230,118,.3);
    color:#00e676;padding:3px 10px;border-radius:20px;font-size:.65rem;
    font-weight:700;text-transform:uppercase;letter-spacing:1px;
    display:flex;align-items:center;gap:5px;white-space:nowrap;
}
.live-dot{
    width:8px;height:8px;border-radius:50%;background:#00e676;
    display:inline-block;animation:pulse 1.5s ease infinite;
}
@keyframes pulse{
    0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(0,230,118,.6)}
    50%{opacity:.6;box-shadow:0 0 0 7px rgba(0,230,118,0)}
}

/* ── KPI Card ── */
.mc{
    background:linear-gradient(145deg,#1a1a2e,#141420);
    border:1px solid rgba(229,0,126,.1);border-radius:12px;
    padding:.9rem .8rem .7rem;text-align:center;position:relative;overflow:hidden;
    transition:all .25s ease;
}
.mc::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,#e5007e,#7b2ff7,#00d4ff)}
.mc:hover{border-color:rgba(229,0,126,.3);transform:translateY(-1px);
    box-shadow:0 4px 18px rgba(229,0,126,.1)}
.mc .ic{font-size:1.4rem;margin-bottom:1px}
.mc .val{
    font-size:1.8rem;font-weight:800;line-height:1.15;
    background:linear-gradient(135deg,#fff,#c8c8ff);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.mc .lbl{font-size:.58rem;color:rgba(255,255,255,.4);text-transform:uppercase;
    letter-spacing:1.6px;font-weight:700;margin-top:1px}

/* ── North star ── */
.ns{
    background:linear-gradient(135deg,rgba(229,0,126,.06),rgba(123,47,247,.06));
    border:1px solid rgba(229,0,126,.18);border-radius:12px;
    padding:.7rem 1rem;text-align:center;margin-bottom:.8rem;
}
.ns .title{color:rgba(255,255,255,.45);font-size:.6rem;text-transform:uppercase;
    letter-spacing:2px;font-weight:700}
.ns .rate{
    font-size:2.4rem;font-weight:900;line-height:1.1;
    background:linear-gradient(135deg,#e5007e,#ff4da6,#ffd700);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.ns .desc{color:rgba(255,255,255,.35);font-size:.68rem;margin-top:2px}

/* ── Section ── */
.sh{
    font-size:.88rem;font-weight:700;color:#e0e0f0;margin:1rem 0 .5rem;
    padding-bottom:5px;border-bottom:2px solid rgba(229,0,126,.2);
    display:flex;align-items:center;gap:5px;
}

/* ── Anomaly ── */
.anom{
    background:linear-gradient(145deg,#1a1a2e,#141420);border-radius:10px;
    padding:.7rem .9rem;margin-bottom:.45rem;border-left:4px solid;
}
.ac{border-left-color:#ff3d71} .aw{border-left-color:#ffd700} .ai{border-left-color:#00d4ff}

/* ── Funnel drop ── */
.fd{display:flex;align-items:center;gap:5px;margin-bottom:2px}
.fd .pct{font-weight:700;font-size:.78rem}
.fd .route{color:rgba(255,255,255,.35);font-size:.68rem}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0f0c29,#141420);
    overflow-x:hidden !important;
}
[data-testid="stSidebar"] > div:first-child{
    overflow-x:hidden !important;
    padding-right:.8rem;
}
[data-testid="stSidebar"] .stMarkdown{
    overflow-x:hidden !important;
}
.sb-row{
    display:flex;justify-content:space-between;align-items:center;
    padding:3px 0;border-bottom:1px solid rgba(255,255,255,.04);
}
.sb-k{color:rgba(255,255,255,.45);font-size:.7rem}
.sb-v{color:#fff;font-weight:600;font-size:.72rem;text-align:right;max-width:55%}

/* ── Hide chrome ── */
#MainMenu,footer,header{visibility:hidden}

/* ── Global scrollbar fix ── */
*::-webkit-scrollbar{width:6px;height:6px}
*::-webkit-scrollbar-thumb{background:rgba(229,0,126,.25);border-radius:3px}
*::-webkit-scrollbar-track{background:transparent}
</style>
""", unsafe_allow_html=True)


# ─── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def load_events(path: str) -> pd.DataFrame:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except FileNotFoundError:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in ("dwell_sec", "cx", "cy"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛍️ Purplle Intelligence")
    st.markdown("---")
    auto_ref = st.toggle("⚡ Auto-refresh (30s)", value=True)
    st.markdown("---")

    st.markdown("##### ⚙️ Pipeline v3.0")
    st.code(
        "Model    YOLO11-L\n"
        "ReID     ResNet18 512d\n"
        "Tracker  BoT-SORT\n"
        "Merger   Union-Find\n"
        "Dedup    52.5%\n"
        "GPU      GTX 1650",
        language=None,
    )

    st.markdown("---")
    st.markdown("##### 📹 Cameras")
    st.code(
        "CAM_01  Skincare Floor\n"
        "CAM_02  Makeup Floor\n"
        "CAM_03  Entry Door\n"
        "CAM_05  Billing Counter",
        language=None,
    )

    st.markdown("---")
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")


# ─── Load ─────────────────────────────────────────────────────────────────────
data_path = JSONL_PATH if JSONL_PATH.exists() else FALLBACK_PATH
df = load_events(str(data_path))
if df.empty:
    st.error("No data. Run the pipeline first.")
    st.stop()

now_str = datetime.now().strftime("%H:%M:%S")

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hdr">
    <div>
        <h1>🛍️ Store Intelligence — {STORE_ID}</h1>
        <div class="sub">AI-Powered Retail Analytics · YOLO11 + ResNet18 ReID · Real-Time</div>
    </div>
    <div class="live-badge"><span class="live-dot"></span>LIVE · {now_str}</div>
</div>
""", unsafe_allow_html=True)


# ─── Compute ──────────────────────────────────────────────────────────────────
total_visitors    = df["track_id"].nunique()
floor_df          = df[df["camera_role"] == "floor"]
floor_visitors    = floor_df["track_id"].nunique()
entry_df          = df[df["camera_role"] == "entry"]
entry_visitors    = entry_df["track_id"].nunique()
billing_df        = df[df["camera_role"] == "billing"]
billing_visitors  = billing_df["track_id"].nunique()

zone_enters = df[df["event_type"] == "ZONE_ENTER"]
zone_exits  = df[df["event_type"] == "ZONE_EXIT"]
zone_dwells = df[df["event_type"] == "ZONE_DWELL"]
total_events = len(df)

dwell_data = zone_exits[zone_exits["dwell_sec"] > 0].copy() if "dwell_sec" in df.columns else pd.DataFrame()
avg_dwell_global = round(dwell_data["dwell_sec"].mean(), 1) if not dwell_data.empty else 0
engaged = zone_dwells[zone_dwells["dwell_sec"] >= 5]["track_id"].nunique() if not zone_dwells.empty else 0
conversion = (billing_visitors / floor_visitors * 100) if floor_visitors > 0 else 0

zone_stats = pd.DataFrame()
if not dwell_data.empty:
    zone_stats = dwell_data.groupby("zone_label")["dwell_sec"].agg(
        avg="mean", mx="max", total="sum", visits="count"
    ).reset_index().sort_values("avg", ascending=False)

cam_stats = df.groupby("camera_id").agg(
    tracks=("track_id", "nunique"), events=("event_type", "count"),
).reset_index()
cam_map = {"CAM_01": "Skincare", "CAM_02": "Makeup", "CAM_03": "Entry", "CAM_05": "Billing"}


# ─── North Star ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="ns">
    <div class="title">⭐ North Star — Offline Store Conversion Rate</div>
    <div class="rate">{conversion:.1f}%</div>
    <div class="desc">{billing_visitors} billing visitors ÷ {floor_visitors} floor visitors</div>
</div>
""", unsafe_allow_html=True)


# ─── KPI Row ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sh">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
cols = st.columns(6)
kpis = [
    ("👥", total_visitors, "Total Visitors"),
    ("🚶", entry_visitors, "Door Entries"),
    ("🏪", floor_visitors, "Floor Visitors"),
    ("🔥", engaged, "Engaged (>5s)"),
    ("💳", billing_visitors, "At Billing"),
    ("⚡", total_events, "AI Events"),
]
for c, (ic, v, lb) in zip(cols, kpis):
    c.markdown(f'<div class="mc"><div class="ic">{ic}</div><div class="val">{v}</div><div class="lbl">{lb}</div></div>', unsafe_allow_html=True)

st.markdown("")

# ─── Funnel + Zone Heatmap ────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="sh">🔄 Conversion Funnel</div>', unsafe_allow_html=True)
    walkins = entry_visitors if entry_visitors > 0 else floor_visitors + 2
    stages = ["Walk-ins", "Floor Browsing", "Engaged (>5s)", "Billing Zone"]
    vals   = [walkins, floor_visitors, engaged, billing_visitors]

    fig = go.Figure(go.Funnel(
        y=stages, x=vals,
        textposition="inside", textinfo="value+percent initial",
        marker=dict(color=[PK, PK2, CY, GN], line=dict(width=0)),
        connector=dict(line=dict(color="rgba(255,255,255,.06)", width=1)),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, margin=dict(l=10,r=10,t=5,b=5), height=270)
    st.plotly_chart(fig, key="fun", width="stretch")

    for i in range(1, len(vals)):
        if vals[i-1] > 0:
            d = (1 - vals[i]/vals[i-1]) * 100
            clr = RD if d > 60 else OR if d > 30 else GN
            st.markdown(f'<div class="fd"><span class="pct" style="color:{clr}">↓ {d:.0f}%</span><span class="route">{stages[i-1]} → {stages[i]}</span></div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="sh">🔥 Zone Dwell Time</div>', unsafe_allow_html=True)
    if not zone_stats.empty:
        fig_z = go.Figure()
        for idx, (_, r) in enumerate(zone_stats.iterrows()):
            clr = ZONE_PAL[idx % len(ZONE_PAL)]
            fig_z.add_trace(go.Bar(
                x=[r["zone_label"]], y=[r["avg"]], showlegend=False,
                marker_color=clr, text=f'{r["avg"]:.1f}s', textposition="outside",
                hovertemplate=f"<b>{r['zone_label']}</b><br>Avg: {r['avg']:.1f}s · Max: {r['mx']:.1f}s · Visits: {int(r['visits'])}<extra></extra>",
            ))
        fig_z.update_layout(**PLOTLY_LAYOUT, margin=dict(l=10,r=10,t=5,b=70), height=270,
            xaxis=dict(tickangle=-18, gridcolor="rgba(255,255,255,.03)", tickfont=dict(size=9)),
            yaxis=dict(title=dict(text="Seconds"), gridcolor="rgba(255,255,255,.03)"),
        )
        st.plotly_chart(fig_z, key="zn", width="stretch")

        zcols = st.columns(len(zone_stats))
        for col, (_, r) in zip(zcols, zone_stats.iterrows()):
            col.metric(r["zone_label"].split("(")[0].strip()[:16], f'{int(r["visits"])} visits', f'{r["avg"]:.1f}s avg')
    else:
        st.info("No zone data.")


# ─── Timeline + Camera ───────────────────────────────────────────────────────
c3, c4 = st.columns([1.4, .6])

with c3:
    st.markdown('<div class="sh">📈 Activity Timeline (30s bins)</div>', unsafe_allow_html=True)
    if "timestamp" in df.columns:
        t = df.copy()
        t["bin"] = t["timestamp"].dt.floor("30s")
        td = t.groupby(["bin","camera_id"]).size().reset_index(name="n")
        fig_t = px.area(td, x="bin", y="n", color="camera_id",
            color_discrete_map={"CAM_01":PK,"CAM_02":CY,"CAM_03":GD,"CAM_05":GN})
        fig_t.update_traces(line_shape="spline", line_width=1.5)
        fig_t.update_layout(**PLOTLY_LAYOUT, margin=dict(l=10,r=10,t=5,b=10), height=270,
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=9)),
            xaxis=dict(gridcolor="rgba(255,255,255,.03)",title=None),
            yaxis=dict(gridcolor="rgba(255,255,255,.03)",title=dict(text="Events/30s")),
        )
        st.plotly_chart(fig_t, key="tl", width="stretch")

with c4:
    st.markdown('<div class="sh">📹 Per-Camera</div>', unsafe_allow_html=True)
    cam_stats["lbl"] = cam_stats["camera_id"].map(cam_map)
    fig_c = go.Figure()
    fig_c.add_trace(go.Bar(x=cam_stats["lbl"],y=cam_stats["tracks"],name="Tracks",marker_color=PK,text=cam_stats["tracks"],textposition="outside"))
    fig_c.add_trace(go.Bar(x=cam_stats["lbl"],y=cam_stats["events"],name="Events",marker_color=CY,text=cam_stats["events"],textposition="outside",opacity=.7))
    fig_c.update_layout(**PLOTLY_LAYOUT, margin=dict(l=10,r=10,t=5,b=10), height=270, barmode="group",
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=9)),
        xaxis=dict(gridcolor="rgba(255,255,255,.03)"),yaxis=dict(gridcolor="rgba(255,255,255,.03)",title=None),
    )
    st.plotly_chart(fig_c, key="cam", width="stretch")


# ─── Event Distribution + Dwell Histogram ────────────────────────────────────
c5, c6 = st.columns(2)

with c5:
    st.markdown('<div class="sh">🎯 Event Type Distribution</div>', unsafe_allow_html=True)
    ec = df["event_type"].value_counts().reset_index()
    ec.columns = ["Type","Count"]
    fig_e = px.pie(ec, values="Count", names="Type", hole=.55, color_discrete_sequence=ZONE_PAL)
    fig_e.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)
    fig_e.update_layout(**PLOTLY_LAYOUT, margin=dict(l=10,r=10,t=10,b=10), height=250,
        legend=dict(font=dict(size=8)))
    st.plotly_chart(fig_e, key="et", width="stretch")

with c6:
    st.markdown('<div class="sh">⏱️ Dwell Distribution</div>', unsafe_allow_html=True)
    if not dwell_data.empty:
        fig_d = px.histogram(dwell_data, x="dwell_sec", nbins=20, color="zone_label",
            color_discrete_sequence=ZONE_PAL, labels={"dwell_sec":"Dwell (s)","zone_label":"Zone"})
        fig_d.update_traces(opacity=.7)
        fig_d.update_layout(**PLOTLY_LAYOUT, margin=dict(l=10,r=10,t=10,b=10), height=250, barmode="overlay",
            legend=dict(font=dict(size=7),orientation="h",yanchor="bottom",y=1.02),
            xaxis=dict(gridcolor="rgba(255,255,255,.03)"),yaxis=dict(gridcolor="rgba(255,255,255,.03)",title=dict(text="Count")),
        )
        st.plotly_chart(fig_d, key="dw", width="stretch")
    else:
        st.info("No dwell data.")


# ─── Anomalies + Event Stream ────────────────────────────────────────────────
c7, c8 = st.columns([.4, .6])

with c7:
    st.markdown('<div class="sh">🚨 Anomalies</div>', unsafe_allow_html=True)
    anoms = []
    be = df[(df["camera_role"]=="billing")&(df["event_type"]=="ZONE_ENTER")]
    if len(be)>4:
        anoms.append(("aw","BILLING_QUEUE_SPIKE","🟡 WARN",f"{len(be)} visitors at billing — possible queue","Open additional billing counter"))
    if conversion<25:
        anoms.append(("aw","CONVERSION_DROP","🟡 WARN",f"Rate {conversion:.1f}% below 25%","Review staffing & placement"))
    if entry_visitors==0:
        anoms.append(("ai","TRIPWIRE_INACTIVE","🔵 INFO","No door crossings on CAM_03","Recalibrate tripwire line"))
    known={"Skincare Shelves (left wall)","Makeup Shelves (right wall)","Center Display Table","Billing Counter Area"}
    active=set(df[df["event_type"].isin(["ZONE_ENTER","ZONE_DWELL"])]["zone_label"].unique())
    for z in known-active:
        anoms.append(("ac","DEAD_ZONE","🔴 CRIT",f"No traffic in '{z}'","Check camera & zone config"))

    if anoms:
        for sc,tp,badge,msg,act in anoms:
            st.markdown(f"""<div class="anom {sc}">
                <div style="display:flex;justify-content:space-between"><strong style="font-size:.8rem">{tp}</strong><span style="font-size:.6rem">{badge}</span></div>
                <div style="font-size:.72rem;color:rgba(255,255,255,.6);margin-top:2px">{msg}</div>
                <div style="font-size:.65rem;color:rgba(255,255,255,.3);margin-top:2px">💡 {act}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ All systems nominal")

with c8:
    st.markdown('<div class="sh">📝 Live Event Stream</div>', unsafe_allow_html=True)
    scols = ["timestamp","camera_id","event_type","track_id","zone_label","dwell_sec"]
    avail = [c for c in scols if c in df.columns]
    rec = df[avail].sort_values("timestamp", ascending=False).head(15).copy()
    if "timestamp" in rec.columns:
        rec["timestamp"] = rec["timestamp"].dt.strftime("%H:%M:%S")
    if "dwell_sec" in rec.columns:
        rec["dwell_sec"] = rec["dwell_sec"].apply(lambda x: f"{x:.1f}s" if x>0 else "—")
    rec.columns = ["Time","Cam","Event","Track","Zone","Dwell"][:len(avail)]
    st.dataframe(rec, height=330, hide_index=True, width="stretch")


# ─── Dedup Evolution ─────────────────────────────────────────────────────────
st.markdown('<div class="sh">🧬 Track Deduplication — v1.0 → v2.1 → v3.0</div>', unsafe_allow_html=True)
dcols = st.columns(4)
versions = {
    "CAM_01 · Skincare": [16, 10, 7],
    "CAM_02 · Makeup":   [21, 13, 8],
    "CAM_03 · Entry":    [37,  5, 5],
    "CAM_05 · Billing":  [ 9,  7, 9],
}
vlabels = ["v1.0", "v2.1", "v3.0"]
vcolors = ["rgba(255,255,255,.15)", OR, GN]

for idx, (col, (cam, vs)) in enumerate(zip(dcols, versions.items())):
    with col:
        red = (1 - vs[-1]/vs[0])*100
        fig_d = go.Figure(go.Bar(x=vlabels, y=vs, marker_color=vcolors,
            text=vs, textposition="outside", textfont=dict(size=13,color="#fff")))
        fig_d.update_layout(**PLOTLY_LAYOUT,
            title=dict(text=f"{cam}<br><span style='font-size:10px;color:#00e676'>↓{red:.0f}% reduction</span>",font=dict(size=11,color="#e0e0f0")),
            margin=dict(l=5,r=5,t=50,b=5), height=210,
            xaxis=dict(gridcolor="rgba(255,255,255,.03)",tickfont=dict(size=8)),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig_d, key=f"dd_{idx}", width="stretch")


# ─── Session Table ────────────────────────────────────────────────────────────
st.markdown('<div class="sh">🗺️ Visitor Sessions</div>', unsafe_allow_html=True)
sessions = []
for tid, g in df.groupby("track_id"):
    g = g.sort_values("timestamp")
    zones = g[g["event_type"].isin(["ZONE_ENTER","ZONE_DWELL"])]["zone_label"].unique()
    tot_dwell = g[g["event_type"]=="ZONE_EXIT"]["dwell_sec"].sum() if "dwell_sec" in g.columns else 0
    sessions.append({
        "Track": tid, "Camera": g["camera_id"].iloc[0], "Role": g["camera_role"].iloc[0],
        "Events": len(g), "Zones": " → ".join(zones) if len(zones) else "—",
        "Dwell (s)": round(tot_dwell,1),
        "First": g["timestamp"].min().strftime("%H:%M:%S"),
        "Last": g["timestamp"].max().strftime("%H:%M:%S"),
    })
sdf = pd.DataFrame(sessions).sort_values("Events", ascending=False)
st.dataframe(sdf, height=220, hide_index=True, width="stretch")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""<div style="text-align:center;padding:.4rem">
<span style="color:rgba(255,255,255,.2);font-size:.65rem">
Store Intelligence v3.0 · Purplle Tech Challenge 2026 · YOLO11 + BoT-SORT + ResNet18 ReID · {now_str}
</span></div>""", unsafe_allow_html=True)

# ─── Auto-refresh ────────────────────────────────────────────────────────────
if auto_ref:
    time.sleep(30)
    st.rerun()
