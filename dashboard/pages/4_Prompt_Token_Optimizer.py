"""Prompt Token Optimizer page: predict a prompt's token/cost before sending it
and trim it with meaning-preserving rewrites."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.prompt_optimizer import (
    available_optimization_rules,
    forecast_prompt_usage,
    optimize_prompt,
    predict_prompt_cost,
)
from dashboard._theme import apply_theme

st.set_page_config(page_title="Prompt Token Optimizer", layout="wide")
apply_theme()
st.title("Prompt Token Optimizer")

st.caption("Predict a prompt's token/cost footprint *before* you send it, then trim it "
           "with meaning-preserving rewrites only — the substantive request is left "
           "unchanged, so output should not be compromised.")

available_rules = available_optimization_rules()
with st.expander(f"Available optimization rules ({len(available_rules)})"):
    for rule in available_rules:
        st.markdown(f"- {rule}")

prompt = st.text_area(
    "Your prompt",
    height=220,
    placeholder="Paste the prompt you're about to send…",
)

st.subheader("Usage assumptions")
usage_col1, usage_col2, usage_col3, usage_col4 = st.columns(4)
with usage_col1:
    expected_output_tokens = st.number_input(
        "Expected output tokens", min_value=0, value=500, step=100,
    )
with usage_col2:
    monthly_requests = st.number_input(
        "Requests per month", min_value=1, value=1000, step=100,
    )
with usage_col3:
    input_price_per_1k = st.number_input(
        "Input price / 1K tokens ($)", min_value=0.0, value=0.003, step=0.001,
        format="%.4f",
    )
with usage_col4:
    output_price_per_1k = st.number_input(
        "Output price / 1K tokens ($)", min_value=0.0, value=0.003, step=0.001,
        format="%.4f",
    )

if st.button("Predict & optimize") and prompt.strip():
    prediction = predict_prompt_cost(prompt, price_per_1k=input_price_per_1k)
    result = optimize_prompt(prompt, price_per_1k=input_price_per_1k)
    forecast = forecast_prompt_usage(
        prompt,
        expected_output_tokens=int(expected_output_tokens),
        monthly_requests=int(monthly_requests),
        input_price_per_1k=input_price_per_1k,
        output_price_per_1k=output_price_per_1k,
    )

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

    st.subheader("Usage cost forecast")
    forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
    forecast_col1.metric(
        "Original monthly cost", f"${forecast['before']['monthly_cost']:,.2f}",
    )
    forecast_col2.metric(
        "Optimized monthly cost", f"${forecast['after']['monthly_cost']:,.2f}",
    )
    forecast_col3.metric(
        "Monthly savings", f"${forecast['monthly_savings']:,.2f}",
    )
    per_request_caption = (
        f"Per request: ${forecast['before']['cost_per_request']:.5f} before, "
        f"${forecast['after']['cost_per_request']:.5f} after. Output cost assumes "
        f"{forecast['expected_output_tokens']:,} tokens per response and is unchanged by "
        "prompt optimization."
    )
    st.caption(per_request_caption.replace("$", r"\$"))

    if result["applied_rules"]:
        st.subheader("Rules applied to this prompt")
        st.dataframe(result["applied_rules"], use_container_width=True, hide_index=True)
    else:
        st.info("No safe reductions found — this prompt is already lean.")

    st.subheader("Assumptions")
    st.json(result["assumptions"])
elif prompt == "":
    st.info("Enter a prompt above to see its predicted token cost and an optimized version.")
