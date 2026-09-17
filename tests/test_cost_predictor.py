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


def test_realistic_estimate_scopes_to_repo(monkeypatch):
    import pandas as pd

    from app import cost_predictor

    df = pd.DataFrame({
        "workspace": ["MyRepo"] * 6 + ["Other"] * 10,
        "prompt_tokens": [100] * 6 + [50000] * 10,
    })
    monkeypatch.setattr("app.prompt_analysis.requests_dataframe", lambda: df)
    est = cost_predictor.realistic_input_estimate("MyRepo")
    assert est["scope"] == "MyRepo"
    assert est["sample_size"] == 6
    assert est["tokens"] == 100


def test_realistic_estimate_falls_back_to_global(monkeypatch):
    import pandas as pd

    from app import cost_predictor

    df = pd.DataFrame({
        "workspace": ["MyRepo"] * 2 + ["Other"] * 10,
        "prompt_tokens": [100] * 2 + [50000] * 10,
    })
    monkeypatch.setattr("app.prompt_analysis.requests_dataframe", lambda: df)
    est = cost_predictor.realistic_input_estimate("MyRepo")
    assert est["scope"] == "all projects"
    assert est["sample_size"] == 12


def test_detect_big_ask_flags_broad_requests():
    from app.cost_predictor import detect_big_ask

    big, terms = detect_big_ask("Please revamp the entire dashboard from scratch")
    assert big is True
    assert terms

    small, terms2 = detect_big_ask("Fix the typo in the header")
    assert small is False
    assert terms2 == []


def test_detect_vague_prompt_flags_unspecific_requests():
    from app.cost_predictor import detect_vague_prompt

    vague, reasons = detect_vague_prompt("fix it")
    assert vague is True
    assert reasons

    vague2, reasons2 = detect_vague_prompt("clean up")
    assert vague2 is True

    concrete, reasons3 = detect_vague_prompt(
        "Fix the off-by-one error in `parse_line()` inside app/ledger.py"
    )
    assert concrete is False
    assert reasons3 == []


def test_cli_warns_on_vague_prompt(monkeypatch, capsys):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    exit_code = main(["fix it", "--output-tokens", "100"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Vague/incomplete prompt (heuristic)" in out


def test_recommend_model_picks_within_tier():
    from app.model_pricing import recommend_model, MODEL_TIERS

    econ_model, econ_cost = recommend_model("economy", 10000, 500)
    assert econ_model in MODEL_TIERS["economy"]
    prem_model, prem_cost = recommend_model("premium", 10000, 500)
    assert prem_model in MODEL_TIERS["premium"]
    # premium should cost at least as much as economy for the same footprint
    assert prem_cost >= econ_cost


def test_recommend_model_falls_back_to_cheapest_for_unknown_tier():
    from app.model_pricing import recommend_model, cheapest_model

    assert recommend_model("nonsense", 1000, 100) == cheapest_model(1000, 100)


def test_cli_prints_heuristic_recommendation(monkeypatch, capsys):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    exit_code = main(["revamp the entire dashboard from scratch", "--output-tokens", "100"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Recommended (heuristic):" in out
    assert "premium tier" in out


def test_expected_output_scales_with_typed_prompt_length_and_big_ask(monkeypatch):
    import pandas as pd

    from app import cost_predictor

    df = pd.DataFrame({
        "workspace": ["MyRepo"] * 8,
        "prompt_text_tokens": [10, 20, 30, 40, 50, 60, 70, 80],
        "output_tokens": [100, 200, 300, 400, 500, 600, 700, 5000],
    })
    monkeypatch.setattr("app.prompt_analysis.requests_dataframe", lambda: df)

    normal = cost_predictor.expected_output_tokens_for(40, repo_name="MyRepo", big_ask=False)
    assert normal["method"] == "typed-length ratio"
    assert normal["scope"] == "MyRepo"

    outlier = cost_predictor.expected_output_tokens_for(5, repo_name="MyRepo", big_ask=True)
    assert outlier["method"].startswith("p90")
    assert outlier["tokens"] > normal["tokens"]


def test_expected_output_falls_back_to_full_input_when_no_typed_text(monkeypatch):
    """Repo has real prompt_tokens but no captured typed text -- should use the
    repo's own full-input ratio rather than jumping straight to global data."""
    import pandas as pd

    from app import cost_predictor

    df = pd.DataFrame({
        "workspace": ["MyRepo"] * 6,
        "prompt_text_tokens": [0] * 6,
        "prompt_tokens": [10000, 20000, 30000, 40000, 50000, 60000],
        "output_tokens": [100, 200, 300, 400, 500, 600],
    })
    monkeypatch.setattr("app.prompt_analysis.requests_dataframe", lambda: df)

    est = cost_predictor.expected_output_tokens_for(5, full_input_tokens=20000, repo_name="MyRepo")
    assert est["scope"] == "MyRepo"
    assert "full-input ratio" in est["method"]


def test_expected_output_ignores_context_inflated_prompt_tokens(monkeypatch):
    """A short typed instruction with huge attached context still gets a small
    output estimate -- context size shouldn't drive the output prediction."""
    import pandas as pd

    from app import cost_predictor

    df = pd.DataFrame({
        "workspace": ["MyRepo"] * 4,
        "prompt_tokens": [200000, 200000, 200000, 200000],  # dominated by attached context
        "prompt_text_tokens": [5, 10, 15, 20],               # what was actually typed
        "output_tokens": [50, 100, 150, 200],
    })
    monkeypatch.setattr("app.prompt_analysis.requests_dataframe", lambda: df)

    est = cost_predictor.expected_output_tokens_for(7, repo_name="MyRepo", big_ask=False)
    assert est["tokens"] < 1000  # would be ~140,000+ if scaled off prompt_tokens instead


def test_realistic_cli_skips_history_when_files_given(tmp_path, capsys):
    ctx = tmp_path / "ctx.py"
    ctx.write_text("print('hi')\n", encoding="utf-8")
    exit_code = main(["Explain this", "--file", str(ctx), "--realistic"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "double-counting" in out
