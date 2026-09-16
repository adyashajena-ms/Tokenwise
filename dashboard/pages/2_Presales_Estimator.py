"""Presales Estimator page: cost interval for a prospective engagement."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os

import plotly.graph_objects as go
import streamlit as st

from app.estimator import estimate_presales_cost
from app.ledger import yield_ledger_by_task_type
from dashboard._theme import apply_theme

st.set_page_config(page_title="Presales Estimator", layout="wide")
apply_theme()
st.title("Presales Estimator")

if not os.path.exists("hack2026.db"):
    st.warning("No dataset found. Go to Home and generate one.")
    st.stop()

task_types = yield_ledger_by_task_type()["task_type"].tolist()
if not task_types:
    st.warning("Dataset is empty. Go to Home and generate one.")
    st.stop()

st.caption("Estimates are a proposed method (empirical bootstrap over historical accepted "
           "outcomes) — not a validated statistical model. Assumptions are shown with every estimate.")

col_a, col_b, col_c = st.columns(3)
with col_a:
    task_type = st.selectbox("Task type", task_types)
with col_b:
    volume = st.number_input("Expected volume (# cases)", min_value=1, value=100, step=10)
with col_c:
    complexity_tier = st.selectbox("Complexity tier", ["none", "low", "medium", "high"])

if st.button("Estimate cost"):
    result = estimate_presales_cost(
        task_type=task_type,
        volume=int(volume),
        complexity_tier=None if complexity_tier == "none" else complexity_tier,
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["P10", "P50 (median)", "P90"],
        y=[result["p10"], result["p50"], result["p90"]],
    ))
    fig.update_layout(title=f"Total cost interval for {volume} x {task_type}", yaxis_title="USD")
    st.plotly_chart(fig, use_container_width=True)

    st.metric("P10", f"${result['p10']:,.2f}")
    st.metric("P50 (median)", f"${result['p50']:,.2f}")
    st.metric("P90", f"${result['p90']:,.2f}")

    st.subheader("Assumptions")
    st.json(result["assumptions"])
