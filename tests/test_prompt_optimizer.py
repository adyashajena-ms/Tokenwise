from app.prompt_optimizer import (
    estimate_cost,
    estimate_tokens,
    optimize_prompt,
    predict_prompt_cost,
)


def test_estimate_tokens_scales_with_length():
    short = estimate_tokens("hello world")
    long = estimate_tokens("hello world " * 50)
    assert short > 0
    assert long > short


def test_empty_prompt_is_zero_tokens():
    assert estimate_tokens("") == 0


def test_predict_reports_tokens_cost_and_method():
    result = predict_prompt_cost("Summarize this report in three bullet points.")
    assert result["tokens"] > 0
    assert result["cost"] == estimate_cost(result["tokens"])
    assert "token_method" in result["assumptions"]


def test_optimize_reduces_tokens_on_filler_heavy_prompt():
    prompt = (
        "Could you please just kindly summarize this really very long report, "
        "and     I was wondering if you could basically list the key points?"
    )
    result = optimize_prompt(prompt)
    assert result["after"]["tokens"] < result["before"]["tokens"]
    assert result["tokens_saved"] > 0
    assert result["percent_saved"] > 0
    assert result["applied_rules"]


def test_optimize_preserves_substantive_content():
    prompt = "Please refactor the login function to use async/await."
    result = optimize_prompt(prompt)
    # The actual instruction terms survive; only politeness is trimmed.
    for keyword in ("refactor", "login", "async/await"):
        assert keyword in result["optimized"]


def test_optimize_is_stable_on_already_lean_prompt():
    prompt = "List three risks."
    result = optimize_prompt(prompt)
    assert result["tokens_saved"] >= 0
    assert result["optimized"].strip() == result["optimized"]
