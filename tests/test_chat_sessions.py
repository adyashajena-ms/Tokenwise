from app.chat_sessions import ChatRequestRecord, _apply_mutation, reconstruct_session
from app.model_pricing import cheapest_model, pricing_for, request_cost
from app.prompt_analysis import (
    cost_by_feature,
    predict_task_cost,
    requests_dataframe,
    savings_if_cheapest_model,
    score_prompts,
    tokens_by_project,
)


def _records():
    return [
        ChatRequestRecord("proj-a", "s1", "Add login", "r1", 1, "copilot/claude-opus-4.6",
                          20000, 400, 40, "Refactor the auth module to async.", 2, 1200,
                          ("auth.py", "login.py")),
        ChatRequestRecord("proj-a", "s1", "Add login", "r2", 2, "copilot/auto",
                          1000, 0, 15, "fix it", 0, 300, ("auth.py",)),
        ChatRequestRecord("proj-b", "s2", "Write tests", "r3", 3, "copilot/gpt-4o-mini",
                          500, 300, 25, "Write a unit test for parse().", 1, 200,
                          ("test_parse.py",)),
    ]


def test_apply_mutation_autovivifies_paths():
    state = {"requests": []}
    _apply_mutation(state, ["requests", 0, "result", "metadata", "promptTokens"], 123)
    assert state["requests"][0]["result"]["metadata"]["promptTokens"] == 123


def test_reconstruct_session_replays_event_log(tmp_path):
    log = tmp_path / "s.jsonl"
    log.write_text(
        '{"kind":0,"v":{"sessionId":"abc","requests":[]}}\n'
        '{"kind":1,"k":["requests",0,"modelId"],"v":"copilot/auto"}\n'
        '{"kind":1,"k":["requests",0,"result","metadata","promptTokens"],"v":50}\n',
        encoding="utf-8",
    )
    state = reconstruct_session(log)
    assert state["sessionId"] == "abc"
    assert state["requests"][0]["result"]["metadata"]["promptTokens"] == 50


def test_cost_by_feature_groups_sessions_with_titles():
    df = requests_dataframe(_records())
    features = cost_by_feature(df, workspace="proj-a")
    assert list(features["feature"]) == ["Add login"]
    row = features.iloc[0]
    assert row["requests"] == 2
    assert row["total_tokens"] == 21400
    assert "auth.py" in row["top_files"]


def test_pricing_and_cheapest_model():
    assert pricing_for("unknown-model") == pricing_for(None)
    cost = request_cost(1000, 1000, "copilot/claude-opus-4.8")
    assert cost > request_cost(1000, 1000, "copilot/claude-sonnet-5")
    model, model_cost = cheapest_model(1000, 1000)
    assert model_cost <= cost


def test_tokens_by_project_aggregates_real_tokens():
    df = requests_dataframe(_records())
    by_project = tokens_by_project(df)
    assert set(by_project["workspace"]) == {"proj-a", "proj-b"}
    proj_a = by_project[by_project["workspace"] == "proj-a"].iloc[0]
    assert proj_a["prompt_tokens"] == 21000
    assert proj_a["total_tokens"] == 21400


def test_score_prompts_flags_wasted_and_good():
    df = requests_dataframe(_records())
    scored = score_prompts(df)
    by_request = scored.set_index("request_id")
    assert by_request.loc["r2", "quality"] == "needs work"
    assert "no output produced" in by_request.loc["r2", "why"]
    assert by_request.loc["r3", "quality"] in {"good", "average"}


def test_savings_if_cheapest_model_non_negative():
    df = requests_dataframe(_records())
    savings = savings_if_cheapest_model(df)
    assert (savings["potential_saving"] >= 0).all()
    opus_row = savings[savings["request_id"] == "r1"].iloc[0]
    assert opus_row["potential_saving"] > 0


def test_predict_task_cost_uses_real_medians():
    df = requests_dataframe(_records())
    prediction = predict_task_cost(df, workspace="proj-a", planned_requests=10)
    assert prediction["planned_requests"] == 10
    assert prediction["predicted_total_cost"] == prediction["median_cost_per_request"] * 10
    assert prediction["assumptions"]["sample_size"] == 2
