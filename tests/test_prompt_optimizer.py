from app.prompt_optimizer import (
    available_optimization_rules,
    estimate_cost,
    estimate_tokens,
    forecast_prompt_usage,
    optimize_prompt,
    predict_prompt_cost,
)


def test_available_rules_exposes_full_optimizer_catalog():
    rules = available_optimization_rules()

    assert len(rules) >= 11
    assert "shorten wordy phrases" in rules
    assert "remove redundant word pairs" in rules
    assert "simplify verbose connectors" in rules
    assert "drop meta commentary" in rules


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


def test_optimize_shortens_wordy_phrases_without_dropping_requirements():
    prompt = (
        "Due to the fact that costs increased, perform an analysis of usage "
        "on a monthly basis and provide an explanation of retry waste prior to launch."
    )
    result = optimize_prompt(prompt)

    assert result["after"]["tokens"] < result["before"]["tokens"]
    assert "because costs increased" in result["optimized"].lower()
    assert "analyze usage monthly" in result["optimized"].lower()
    assert "explain retry waste before launch" in result["optimized"].lower()
    assert any(rule["rule"] == "shorten wordy phrases" for rule in result["applied_rules"])


def test_optimize_removes_redundancy_and_verbose_connectors():
    prompt = (
        "It is important to note that each and every support case must use the exact same "
        "template at all times. On a weekly basis, review the end result by means of the ledger."
    )
    result = optimize_prompt(prompt)
    optimized = result["optimized"].lower()

    assert result["after"]["tokens"] < result["before"]["tokens"]
    assert "each support case" in optimized
    assert "same template always" in optimized
    assert "weekly, review the result using the ledger" in optimized
    assert {rule["rule"] for rule in result["applied_rules"]} >= {
        "remove redundant word pairs",
        "simplify verbose connectors",
        "drop meta commentary",
    }


def test_optimize_is_stable_on_already_lean_prompt():
    prompt = "List three risks."
    result = optimize_prompt(prompt)
    assert result["tokens_saved"] >= 0
    assert result["optimized"].strip() == result["optimized"]


def test_forecast_includes_output_cost_and_monthly_volume():
    prompt = "Could you please summarize this report?"
    result = forecast_prompt_usage(
        prompt,
        expected_output_tokens=500,
        monthly_requests=100,
        input_price_per_1k=0.01,
        output_price_per_1k=0.03,
    )

    output_cost = estimate_cost(500, 0.03)
    expected_before = estimate_cost(estimate_tokens(prompt), 0.01) + output_cost
    assert result["before"]["cost_per_request"] == expected_before
    assert result["before"]["monthly_cost"] == expected_before * 100
    assert result["after"]["monthly_cost"] <= result["before"]["monthly_cost"]
    assert result["monthly_savings"] >= 0


def test_forecast_rejects_invalid_planning_inputs():
    try:
        forecast_prompt_usage("Summarize this.", -1, 100)
    except ValueError as error:
        assert "expected_output_tokens" in str(error)
    else:
        raise AssertionError("negative expected output tokens should fail")
