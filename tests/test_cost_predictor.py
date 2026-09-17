import pytest

from app.cost_predictor import gather_git_context_files, predict_across_models
from app.model_pricing import MODEL_PRICING
from app.predict import main


def test_predict_across_models_covers_all_models():
    result = predict_across_models("Refactor the auth module.", expected_output_tokens=500)
    assert result["prompt_tokens"] > 0
    assert len(result["models"]) == len(MODEL_PRICING)
    # sorted ascending by total cost
    costs = [m["total_cost"] for m in result["models"]]
    assert costs == sorted(costs)
    assert result["cheapest"]["total_cost"] <= result["most_expensive"]["total_cost"]


def test_context_files_add_input_tokens(tmp_path):
    ctx = tmp_path / "ctx.py"
    ctx.write_text("def add(a, b):\n    return a + b\n" * 20, encoding="utf-8")
    with_ctx = predict_across_models("Explain", [str(ctx)], 300)
    without_ctx = predict_across_models("Explain", [], 300)
    assert with_ctx["context_tokens"] > 0
    assert with_ctx["input_tokens"] > without_ctx["input_tokens"]


def test_missing_file_reports_error_not_crash():
    result = predict_across_models("Explain", ["does_not_exist_123.py"], 100)
    assert result["file_breakdown"][0]["error"]
    assert result["context_tokens"] == 0


def test_negative_output_tokens_rejected():
    with pytest.raises(ValueError):
        predict_across_models("hi", expected_output_tokens=-5)


def test_cli_runs_with_inline_prompt(capsys):
    exit_code = main(["Write a unit test", "--output-tokens", "400"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "MODEL" in out and "TOTAL cr" in out
    assert "Cheapest:" in out


def test_gather_git_context_files_returns_code_files():
    files = gather_git_context_files(".")
    # runs read-only git; every returned path must be a real code file
    assert isinstance(files, list)
    for path in files:
        assert path.rsplit(".", 1)[-1] in {
            "py", "ts", "tsx", "js", "jsx", "cs", "java", "go", "rb", "rs",
            "cpp", "c", "h", "hpp", "xaml", "json", "yaml", "yml", "md", "sql",
        }


def test_cli_auto_mode_runs(capsys):
    exit_code = main(["Add a feature", "--auto"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Auto mode:" in out
    assert "TOTAL cr" in out


def test_extra_context_tokens_increase_input():
    base = predict_across_models("hi", [], 100)
    boosted = predict_across_models("hi", [], 100, extra_context_tokens=100000)
    assert boosted["input_tokens"] == base["input_tokens"] + 100000
    assert boosted["cheapest"]["total_cost"] > base["cheapest"]["total_cost"]


def test_long_context_tier_present_for_gpt56_models():
    from app.model_pricing import long_context_pricing_for

    sol_lc = long_context_pricing_for("copilot/gpt-5.6-sol")
    assert sol_lc == {"input": 800, "output": 3000, "cache_read": 80, "cache_write": 1000}
    terra_lc = long_context_pricing_for("copilot/gpt-5.6-terra")
    assert terra_lc["output"] == 1800
    # models without a long-context tier return None
    assert long_context_pricing_for("copilot/claude-opus-4.8") is None
