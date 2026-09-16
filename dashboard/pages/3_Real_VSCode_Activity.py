"""Real VS Code Activity page: ingest local Copilot session transcripts and
show cost/rework for actual coding sessions, using the same ledger engine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os

import plotly.express as px
import streamlit as st

from app.ingest_vscode_sessions import TASK_TYPE_PREFIX, ingest_vscode_sessions
from app.ledger import trace_case, yield_ledger_by_task_type
from dashboard._theme import apply_theme

st.set_page_config(page_title="Real VS Code Activity", layout="wide")
apply_theme()
st.title("Real VS Code Activity")

st.caption(
    "Reads local Copilot chat transcripts from every VS Code workspace on this machine "
    "(nothing leaves your machine). Cost is a PROXY estimate (~chars/4 tokens x an assumed "
    "$/1K-token rate) since local transcripts don't record actual billed token counts -- "
    "treat it as directional, not exact. 'Rework' = the same file edited more than once "
    "within a single request."
)

use_git_signal = st.checkbox(
    "Use git-diff signal for accepted/abandoned (experimental)",
    value=False,
    help="Instead of inferring acceptance from transcript shape, run read-only git checks "
         "to see whether each request's edited files were committed at/after the request. "
         "File-level, directional proxy — uncommitted work reads as not kept.",
)

if st.button("Scan my VS Code Copilot sessions"):
    with st.spinner("Scanning local transcripts..."):
        stats = ingest_vscode_sessions(use_git_signal=use_git_signal)
    st.success(
        f"Ingested {stats['workspaces']} workspace(s), {stats['cases']} request(s), "
        f"{stats['attempts']} attempt(s). Skipped {stats['skipped_files']} unreadable file(s)."
    )
    if use_git_signal:
        st.info(
            f"Git signal: {stats['git_scored_cases']} of {stats['cases']} request(s) scored "
            f"from local git history ({stats['git_kept_cases']} judged kept). The rest fell "
            "back to the transcript-shape heuristic. File-level, directional proxy — uncommitted "
            "work reads as not kept."
        )

if not os.path.exists("hack2026.db"):
    st.info("No dataset yet. Click the button above to scan.")
    st.stop()

ledger_df = yield_ledger_by_task_type()
real_df = ledger_df[ledger_df["task_type"].str.startswith(TASK_TYPE_PREFIX)].copy()
if real_df.empty:
    st.info("No real session data ingested yet. Click the button above to scan.")
    st.stop()

real_df["project"] = real_df["task_type"].str.removeprefix(TASK_TYPE_PREFIX)

st.subheader("Cost per accepted request, by project")
st.dataframe(
    real_df[["project", "total_spend", "accepted_outcomes", "abandoned_cases",
             "cost_per_accepted_outcome", "yield_ratio", "retry_waste", "rework_waste",
             "git_verified_outcomes"]]
    .style.format({
        "total_spend": "${:,.4f}",
        "cost_per_accepted_outcome": "${:,.4f}",
        "yield_ratio": "{:.1%}",
        "retry_waste": "${:,.4f}",
        "rework_waste": "${:,.4f}",
    }),
    use_container_width=True,
)
st.caption("`git_verified_outcomes` = requests whose accepted/abandoned status was decided "
           "from local git history (0 unless you scanned with the git-diff signal enabled).")

fig = px.bar(
    real_df.melt(id_vars="project", value_vars=["retry_waste", "rework_waste", "abandoned_waste"],
                 var_name="waste_type", value_name="amount"),
    x="project", y="amount", color="waste_type", barmode="stack",
    title="Where the (proxy) cost went: retries vs. rework vs. abandoned requests",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Drill down: trace one request back to its attempts")
case_id = st.number_input("Case ID", min_value=1, step=1, value=1)
if st.button("Trace request"):
    trace_df = trace_case(int(case_id))
    if trace_df.empty:
        st.error(f"No case found with id={case_id}")
    else:
        st.dataframe(trace_df, use_container_width=True)
