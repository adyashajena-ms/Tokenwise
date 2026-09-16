"""Prompt Token Optimizer page: predict a prompt's token/cost before sending it
and trim it with meaning-preserving rewrites."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.prompt_optimizer import optimize_prompt, predict_prompt_cost
from dashboard._theme import apply_theme

st.set_page_config(page_title="Prompt Token Optimizer", layout="wide")
apply_theme()
st.title("Prompt Token Optimizer")

st.caption("Predict a prompt's token/cost footprint *before* you send it, then trim it "
           "with meaning-preserving rewrites only — the substantive request is left "
           "unchanged, so output should not be compromised.")

prompt = st.text_area(
    "Your prompt",
    height=220,
    placeholder="Paste the prompt you're about to send…",
)

if st.button("Predict & optimize") and prompt.strip():
    prediction = predict_prompt_cost(prompt)
    result = optimize_prompt(prompt)

    st.subheader("Prediction (before sending)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted tokens", f"{prediction['tokens']:,}")
    c2.metric("Predicted cost", f"${prediction['cost']:.5f}")
    c3.metric("Characters", f"{prediction['characters']:,}")
    st.caption(f"Token method: {prediction['assumptions']['token_method']}")

    st.subheader("Optimized prompt")
    c4, c5, c6 = st.columns(3)
    c4.metric("Tokens", f"{result['after']['tokens']:,}",
              delta=f"-{result['tokens_saved']:,}", delta_color="inverse")
    c5.metric("Cost", f"${result['after']['cost']:.5f}",
              delta=f"-${result['cost_saved']:.5f}", delta_color="inverse")
    c6.metric("Saved", f"{result['percent_saved']:.1f}%")

    st.text_area("Optimized (review before use)", value=result["optimized"], height=180)

    if result["applied_rules"]:
        st.subheader("What was changed")
        st.dataframe(result["applied_rules"], use_container_width=True, hide_index=True)
    else:
        st.info("No safe reductions found — this prompt is already lean.")

    st.subheader("Assumptions")
    st.json(result["assumptions"])
elif prompt == "":
    st.info("Enter a prompt above to see its predicted token cost and an optimized version.")
