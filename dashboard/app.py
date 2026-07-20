"""
Real-time fraud scoring dashboard for Project 6.

Reads the predictions written by scoring/score_stream.py (Bronze
transactions augmented with fraud_probability and predicted_fraud) and
renders a live model-quality view: not just "what happened" (Project 1's
dashboard), but "how well is the model doing right now" -- rolling
precision/recall, a dedicated alerts feed with a correct/incorrect
indicator against ground truth, and the probability score over time.

This is possible here specifically because Project 1's producer injects
ground-truth fraud labels into every simulated transaction -- most
real-time fraud dashboards can't show live accuracy, because they don't
have an oracle to check predictions against.

Local runs read the live data/predictions/ folder written by the scoring
job. When deployed (e.g. Streamlit Community Cloud, which runs neither
Spark nor Docker), it falls back to a small versioned sample at
data/predictions_sample/predictions_sample.parquet -- see
scoring/export_sample.py.

Run (inside the `fraud-ml` conda env):
    streamlit run dashboard/app.py
"""

import glob
import json
import os
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "data", "predictions")
SAMPLE_PATH = os.path.join(PROJECT_ROOT, "data", "predictions_sample", "predictions_sample.parquet")
METRICS_PATH = os.path.join(PROJECT_ROOT, "training", "metrics.json")

ROLLING_WINDOW_MINUTES = 10  # window for "live" precision/recall

# ---------------------------------------------------------------------------
# Palette -- same "operations console" theme as Project 1's dashboard, kept
# consistent since this is a continuation of the same pipeline.
# ---------------------------------------------------------------------------
INK = "#0d1117"
PANEL = "#161b22"
GRID = "#21262d"
TEXT = "#c9d1d9"
MUTED = "#6e7681"
AMBER = "#d29922"
RED = "#f85149"
GREEN = "#3fb950"
BLUE = "#58a6ff"

st.set_page_config(
    page_title="Fraud Scoring Monitor",
    page_icon="•",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background: {INK}; color: {TEXT}; }}
    .block-container {{ padding-top: 3.5rem; padding-bottom: 4rem; max-width: 1300px; }}
    h1, h2, h3 {{ color: {TEXT}; font-family: 'Inter', system-ui, sans-serif; letter-spacing: -0.01em; }}
    .eyebrow {{
        font-family: ui-monospace, 'SFMono-Regular', Menlo, monospace;
        font-size: 0.72rem; letter-spacing: 0.22em; text-transform: uppercase;
        color: {AMBER}; margin-bottom: 0.35rem;
    }}
    .kpi {{
        background: {PANEL}; border: 1px solid {GRID}; border-radius: 10px;
        padding: 1.1rem 1.3rem;
    }}
    .kpi .label {{
        font-family: ui-monospace, monospace; font-size: 0.7rem;
        letter-spacing: 0.14em; text-transform: uppercase; color: {MUTED};
    }}
    .kpi .value {{
        font-family: ui-monospace, monospace; font-size: 2.0rem;
        font-weight: 600; color: {TEXT}; line-height: 1.25; margin-top: 0.2rem;
    }}
    .kpi .value.alert {{ color: {RED}; }}
    .kpi .value.warn {{ color: {AMBER}; }}
    .kpi .value.good {{ color: {GREEN}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=5)
def load_predictions() -> tuple:
    # Local: read the full predictions folder written by score_stream.py.
    # Deploy (no Spark/Kafka running): fall back to the small versioned sample.
    files = glob.glob(os.path.join(PREDICTIONS_DIR, "*.parquet"))
    if files:
        frames = []
        for f in files:
            try:
                frames.append(pd.read_parquet(f))
            except Exception:
                continue
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df = df.drop_duplicates(subset=["transaction_id"])
            return df.sort_values("event_time").reset_index(drop=True), "live"

    if os.path.exists(SAMPLE_PATH):
        df = pd.read_parquet(SAMPLE_PATH)
        df = df.drop_duplicates(subset=["transaction_id"])
        return df.sort_values("event_time").reset_index(drop=True), "sample"

    return pd.DataFrame(), "none"


@st.cache_data(ttl=30)
def load_offline_metrics() -> dict:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


def kpi(col, label, value, kind=""):
    col.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value {kind}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def precision_recall(sub: pd.DataFrame) -> tuple:
    tp = int((sub["predicted_fraud"] & sub["is_fraud"]).sum())
    fp = int((sub["predicted_fraud"] & ~sub["is_fraud"]).sum())
    fn = int((~sub["predicted_fraud"] & sub["is_fraud"]).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    return precision, recall, tp, fp, fn


# ---------------------------------------------------------------------------
# Header + controls
# ---------------------------------------------------------------------------
left, right = st.columns([3, 1])
with left:
    st.markdown('<div class="eyebrow">Real-time model scoring</div>', unsafe_allow_html=True)
    st.markdown("# Fraud Scoring Monitor")
with right:
    auto = st.toggle("Auto-refresh (5s)", value=False)

df, source = load_predictions()
offline_metrics = load_offline_metrics()

if df.empty:
    st.info(
        "No predictions yet. Start the pipeline: run producer/producer.py "
        "(Project 1) and scoring/score_stream.py, then this panel will fill in.",
    )
    st.stop()

if source == "live":
    st.caption("Reading live predictions written by score_stream.py.")
else:
    st.caption(
        "Reading a versioned sample (data/predictions_sample/) -- no live Spark/Kafka "
        "pipeline running in this environment. Run the full local pipeline for live data.",
    )

# ---------------------------------------------------------------------------
# KPIs -- overall + rolling precision/recall
# ---------------------------------------------------------------------------
precision_all, recall_all, tp_all, fp_all, fn_all = precision_recall(df)

window_start = df["event_time"].max() - pd.Timedelta(minutes=ROLLING_WINDOW_MINUTES)
rolling = df[df["event_time"] >= window_start]
precision_roll, recall_roll, tp_roll, fp_roll, fn_roll = precision_recall(rolling)

c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "Transactions scored", f"{len(df):,}")
kpi(c2, "Alerts raised", f"{int(df['predicted_fraud'].sum()):,}", "warn")
kpi(
    c3, f"Precision (last {ROLLING_WINDOW_MINUTES}m)",
    f"{precision_roll:.1%}" if precision_roll is not None else "—",
    "good" if (precision_roll or 0) >= 0.9 else "alert",
)
kpi(
    c4, f"Recall (last {ROLLING_WINDOW_MINUTES}m)",
    f"{recall_roll:.1%}" if recall_roll is not None else "—",
    "good" if (recall_roll or 0) >= 0.8 else "warn",
)
kpi(c5, "False positives / negatives", f"{fp_roll} / {fn_roll}", "alert" if fn_roll > 0 else "")

st.caption(
    f"Rolling window: last {ROLLING_WINDOW_MINUTES} min of transaction "
    f"time · Overall so far: precision {precision_all:.1%} · recall "
    f"{recall_all:.1%}" if precision_all is not None and recall_all is not None else ""
)

st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Fraud probability over time
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Model output</div>', unsafe_allow_html=True)
st.markdown("### Fraud probability over time")
threshold = offline_metrics.get("threshold", 0.5)

prob_fig = go.Figure()
legit = df[~df["is_fraud"]]
fraud = df[df["is_fraud"]]
prob_fig.add_trace(go.Scattergl(
    x=legit["event_time"], y=legit["fraud_probability"], mode="markers",
    marker=dict(size=4, color=BLUE, opacity=0.4), name="Legitimate",
))
prob_fig.add_trace(go.Scattergl(
    x=fraud["event_time"], y=fraud["fraud_probability"], mode="markers",
    marker=dict(size=6, color=RED, opacity=0.85), name="Actual fraud",
))
prob_fig.add_hline(y=threshold, line_dash="dash", line_color=AMBER,
                    annotation_text=f"Alert threshold ({threshold:.3f})",
                    annotation_font_color=AMBER)
prob_fig.update_layout(
    height=340, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    margin=dict(l=10, r=10, t=10, b=10), font=dict(color=TEXT),
    xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID, range=[0, 1]),
    legend=dict(orientation="h", y=1.1, x=0),
)
st.plotly_chart(prob_fig, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})

# ---------------------------------------------------------------------------
# Recall by fraud type -- proves the model catches all 3 patterns live,
# not just the easy, high-volume one. Mirrors the offline metric from
# training/metrics.json so the two can be compared directly.
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Detection quality</div>', unsafe_allow_html=True)
st.markdown("### Recall by fraud type — live vs. offline (training)")

fraud_only = df[df["is_fraud"]]
live_recall_by_type = {}
for ftype in ["high_amount", "card_testing", "impossible_travel"]:
    subset = fraud_only[fraud_only["fraud_type"] == ftype]
    live_recall_by_type[ftype] = float(subset["predicted_fraud"].mean()) if len(subset) > 0 else None

offline_recall_by_type = offline_metrics.get("recall_by_fraud_type", {})

type_fig = go.Figure()
labels = ["high_amount", "card_testing", "impossible_travel"]
type_fig.add_trace(go.Bar(
    x=labels, y=[offline_recall_by_type.get(t) for t in labels],
    name="Offline (training test set)", marker_color=MUTED,
))
type_fig.add_trace(go.Bar(
    x=labels, y=[live_recall_by_type.get(t) for t in labels],
    name="Live (this stream)", marker_color=GREEN,
))
type_fig.update_layout(
    barmode="group", height=300, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    margin=dict(l=10, r=10, t=10, b=10), font=dict(color=TEXT),
    xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID, range=[0, 1]),
    legend=dict(orientation="h", y=1.15, x=0),
)
st.plotly_chart(type_fig, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
st.caption(
    "Live recall on a short-running local demo is noisy (few examples per "
    "type) -- shown alongside the offline test-set numbers for context, not "
    "as a replacement for them."
)

# ---------------------------------------------------------------------------
# Alerts feed -- only flagged transactions, ranked by probability, with a
# correct/incorrect indicator against ground truth.
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Alerts</div>', unsafe_allow_html=True)
st.markdown("### Flagged transactions")

alerts = df[df["predicted_fraud"]].sort_values("fraud_probability", ascending=False).head(25).copy()
alerts["Correct"] = alerts["is_fraud"].map(lambda x: "✅" if x else "❌")
alerts["fraud_probability"] = alerts["fraud_probability"].round(3)
alerts["amount"] = alerts["amount"].round(2)

show_alerts = alerts[[
    "event_time", "card_id", "amount", "category", "city",
    "fraud_probability", "fraud_type", "Correct",
]].rename(columns={
    "event_time": "Time", "card_id": "Card", "amount": "Amount $",
    "category": "Category", "city": "City",
    "fraud_probability": "Score", "fraud_type": "Actual type",
})
show_alerts["Actual type"] = show_alerts["Actual type"].fillna("— (false positive)")

st.dataframe(
    show_alerts.style.format({"Amount $": "${:,.2f}", "Score": "{:.3f}"}),
    use_container_width=True, hide_index=True,
)

st.caption(
    f"{len(df):,} transactions scored · pipeline: producer → Redpanda → "
    f"Spark Structured Streaming (foreachBatch) → XGBoost → Parquet → this dashboard",
)

if auto:
    time.sleep(5)
    st.rerun()
