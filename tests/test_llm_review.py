from unittest.mock import Mock, patch

from app.llm_review import llm_available, review_prompt_with_llm
from app.predict import main


def test_llm_available_false_without_env(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    assert llm_available() is False


def test_llm_available_true_with_all_env_set(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    assert llm_available() is True


def test_review_prompt_with_llm_parses_response(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    fake_response = Mock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "choices": [{"message": {"content":
            '{"vague": true, "reasons": ["no target named"], '
            '"rewritten_prompt": "Fix the null check in app/api.py"}'
        }}]
    }
    with patch("app.llm_review.requests.post", return_value=fake_response) as mock_post:
        result = review_prompt_with_llm("fix it")
        # the API key must never appear in the URL (only in headers)
        assert "test-key" not in mock_post.call_args.args[0]

    assert result["vague"] is True
    assert result["rewritten_prompt"] == "Fix the null check in app/api.py"


def test_review_prompt_with_llm_returns_none_on_network_error(monkeypatch):
    import requests

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    with patch("app.llm_review.requests.post", side_effect=requests.ConnectionError):
        assert review_prompt_with_llm("fix it") is None


def test_v1_endpoint_uses_bearer_and_chat_completions_path(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT",
                       "https://tokenwise.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

    from app.llm_review import _build_request_target

    url, headers = _build_request_target()
    assert url == "https://tokenwise.services.ai.azure.com/openai/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert "api-key" not in headers


def test_foundry_project_endpoint_uses_api_key_and_v1_path(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT",
                       "https://tokenwise.services.ai.azure.com/api/projects/tokenwise")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

    from app.llm_review import _build_request_target

    url, headers = _build_request_target()
    assert url == ("https://tokenwise.services.ai.azure.com/api/projects/tokenwise"
                   "/openai/v1/chat/completions")
    assert headers["api-key"] == "test-key"
    assert "Authorization" not in headers


def test_classic_endpoint_uses_api_key_header(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://tokenwise.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    from app.llm_review import _build_request_target

    url, headers = _build_request_target()
    assert "/openai/deployments/gpt-4o-mini/chat/completions" in url
    assert "api-version=" in url
    assert headers["api-key"] == "test-key"


def test_review_prompt_with_llm_none_without_credentials(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    assert review_prompt_with_llm("fix it") is None


def test_cli_uses_llm_rewrite_when_configured(monkeypatch, capsys):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    monkeypatch.setattr(
        "app.predict.review_prompt_with_llm",
        lambda prompt: {"vague": True, "reasons": ["too vague"],
                         "rewritten_prompt": "Fix the null check in app/api.py"},
    )
    exit_code = main(["fix it", "--output-tokens", "100"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "LLM-reviewed" in out
    assert "Suggested rewrite: Fix the null check in app/api.py" in out


def test_cli_falls_back_to_heuristic_without_llm_config(monkeypatch, capsys):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    exit_code = main(["fix it", "--output-tokens", "100"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "heuristic" in out
