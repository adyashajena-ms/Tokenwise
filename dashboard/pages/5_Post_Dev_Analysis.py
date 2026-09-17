"""Post-Dev Analysis: REAL token usage from VS Code chatSessions.

Shows which project folder spent how many tokens, classifies your prompts as
good / average / needs work so you can learn, and estimates what switching to a
cheaper model would have saved.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px
import streamlit as st

from app.prompt_analysis import (
    cost_by_feature,
    requests_dataframe,
    savings_if_cheapest_model,
    score_prompts,
    tokens_by_project,
)
from dashboard._theme import apply_theme

st.set_page_config(page_title="Post-Dev Analysis", layout="wide")
apply_theme()
st.title("Post-Dev Analysis")
st.caption(
    "Real prompt/output token counts read locally from VS Code chatSessions "
    "(nothing leaves your machine). Cost is in Copilot credits (per 1M tokens), "
    "the same units shown in the model picker."
)


@st.cache_data(show_spinner="Reading real chatSessions token usage…")
def _load():
    return requests_dataframe()


if st.button("Scan / refresh chatSessions"):
    _load.clear()

df = _load()
if df.empty:
    st.info("No chatSessions found on this machine yet.")
    st.stop()

total_prompt = int(df["prompt_tokens"].sum())
total_output = int(df["output_tokens"].sum())
total_cost = float(df["cost"].sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Requests", f"{len(df):,}")
m2.metric("Prompt tokens", f"{total_prompt:,}")
m3.metric("Output tokens", f"{total_output:,}")
m4.metric("Est. credits", f"{total_cost:,.0f}")

st.subheader("Tokens & credits by project folder")
by_project = tokens_by_project(df)
st.dataframe(
    by_project.style.format({
        "prompt_tokens": "{:,}",
        "output_tokens": "{:,}",
        "total_tokens": "{:,}",
        "cost": "{:,.2f} cr",
    }),
    use_container_width=True,
)
fig = px.bar(
    by_project, x="workspace", y="total_tokens",
    title="Real total tokens by project", text_auto=".2s",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Credits by feature (per chat session)")
project_choice = st.selectbox(
    "Project", by_project["workspace"].tolist(),
    help="Each chat session is treated as one feature. The label is the VS Code "
         "session title (a hint that can be off) — use the 'top_files' column, "
         "which shows the actual files touched, as the grounded signal.",
)
features = cost_by_feature(df, workspace=project_choice)
if features.empty:
    st.info("No feature-level sessions found for this project.")
else:
    st.dataframe(
        features.style.format({
            "prompt_tokens": "{:,}",
            "output_tokens": "{:,}",
            "total_tokens": "{:,}",
            "cost": "{:,.2f} cr",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "`feature` is the VS Code session title (auto-generated, may mislead). "
        "`top_files` lists the most-edited files in that session — the concrete "
        "evidence of what the feature actually was. `cost` is in Copilot credits."
    )
    st.plotly_chart(
        px.bar(features.head(15), x="cost", y="feature", orientation="h",
               title=f"Top features by credits — {project_choice}"),
        use_container_width=True,
    )


st.subheader("Learn from your prompts: good vs. needs work")
scored = score_prompts(df)
if scored.empty:
    st.info("No prompts with text found to score.")
else:
    quality_counts = scored["quality"].value_counts().reset_index()
    quality_counts.columns = ["quality", "count"]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(
            px.pie(quality_counts, names="quality", values="count",
                   title="Prompt quality mix"),
            use_container_width=True,
        )
    with c2:
        st.dataframe(
            scored[["workspace", "quality", "why", "prompt_tokens",
                    "output_tokens", "cost", "prompt_text"]]
            .head(50)
            .style.format({
                "prompt_tokens": "{:,}",
                "output_tokens": "{:,}",
                "cost": "{:,.2f} cr",
            }),
            use_container_width=True,
            hide_index=True,
        )

st.subheader("Would a cheaper model have helped?")
savings = savings_if_cheapest_model(df)
by_saving = (
    savings.groupby("model_id")
    .agg(requests=("request_id", "count"),
         cost=("cost", "sum"),
         potential_saving=("potential_saving", "sum"))
    .reset_index()
    .sort_values("potential_saving", ascending=False)
)
st.dataframe(
    by_saving.style.format({
        "cost": "{:,.2f} cr",
        "potential_saving": "{:,.2f} cr",
    }),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "Credits saved = cost at the model used minus cost of the cheapest known "
    "model for the same token footprint. Quality trade-offs are not modeled — "
    "treat as an upper bound worth reviewing."
)
