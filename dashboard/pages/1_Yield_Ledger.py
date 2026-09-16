"""Yield Ledger page: cost per accepted outcome + waste breakdown + drill-down."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os

import plotly.express as px
import streamlit as st

from app.ledger import trace_case, yield_ledger_by_task_type
from dashboard._theme import apply_theme

st.set_page_config(page_title="Yield Ledger", layout="wide")
apply_theme()
st.title("Yield Ledger")

if not os.path.exists("hack2026.db"):
    st.warning("No dataset found. Go to Home and generate one.")
    st.stop()

ledger_df = yield_ledger_by_task_type()
if ledger_df.empty:
    st.warning("Dataset is empty. Go to Home and generate one.")
    st.stop()

st.subheader("Cost per accepted outcome, by task type")
st.dataframe(
    ledger_df.style.format({
        "total_spend": "${:,.2f}",
        "cost_per_accepted_outcome": "${:,.2f}",
        "yield_ratio": "{:.1%}",
        "retry_waste": "${:,.2f}",
        "rework_waste": "${:,.2f}",
        "abandoned_waste": "${:,.2f}",
    }),
    use_container_width=True,
)

col1, col2 = st.columns(2)
with col1:
    fig_cost = px.bar(
        ledger_df, x="task_type", y="cost_per_accepted_outcome",
        title="Cost per accepted outcome", text_auto=".2s",
    )
    st.plotly_chart(fig_cost, use_container_width=True)

with col2:
    waste_long = ledger_df.melt(
        id_vars="task_type",
        value_vars=["retry_waste", "rework_waste", "abandoned_waste"],
        var_name="waste_type", value_name="amount",
    )
    fig_waste = px.bar(
        waste_long, x="task_type", y="amount", color="waste_type",
        title="Waste breakdown (spend not tied to a winning attempt)", barmode="stack",
    )
    st.plotly_chart(fig_waste, use_container_width=True)

st.subheader("Drill down: trace an accepted outcome back to its spend events")
case_id = st.number_input("Case ID", min_value=1, step=1, value=1)
if st.button("Trace case"):
    trace_df = trace_case(int(case_id))
    if trace_df.empty:
        st.error(f"No case found with id={case_id}")
    else:
        st.dataframe(trace_df, use_container_width=True)
