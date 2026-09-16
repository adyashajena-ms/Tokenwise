"""Streamlit entry page: dataset setup and challenge overview."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

import streamlit as st

from app.data_gen import generate_dataset
from dashboard._theme import apply_theme

st.set_page_config(page_title="TokenWise", layout="wide")
apply_theme()

st.title("TokenWise")
st.caption("InSpireD Executive Challenge — cost per accepted outcome, presales cost intervals, "
           "and token optimization before you spend")

st.markdown(
    """
This prototype connects AI spend to accepted work — and cuts waste before it happens:

- **Yield Ledger** — cost per accepted outcome, traced back to raw spend events, with a
  waste breakdown (retries / rework / abandonment).
- **Presales Estimator** — a defensible cost interval (P10/P50/P90) for a new engagement,
  built from historical accepted-outcome costs, with its assumptions stated alongside it.
- **Prompt Token Optimizer** — predicts a prompt's token count and cost *before* you send
  it, then trims it with meaning-preserving rewrites so fewer tokens are spent without
  compromising output.

Data below is **synthetic** (three task-type archetypes with different cost/retry/abandon
profiles) so the ledger and estimator have realistic variance to show. Use the sidebar to
open any of the pages.
"""
)

db_exists = os.path.exists("hack2026.db")
st.subheader("Dataset")
st.write("Database found." if db_exists else "No database found yet — generate one below.")

cases_per_archetype = st.slider("Cases per task-type archetype", 30, 500, 150, step=10)
if st.button("(Re)generate synthetic dataset"):
    if db_exists:
        os.remove("hack2026.db")
    generate_dataset(cases_per_archetype=cases_per_archetype)
    st.success("Synthetic dataset generated. Open the Yield Ledger or Presales Estimator pages.")
