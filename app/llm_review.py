"""Optional LLM-based review of a prompt for vagueness/incompleteness.

Uses Azure OpenAI if configured via environment variables -- NEVER pass the key
as a CLI argument or paste it into chat; set it directly in your own shell/env.

Two endpoint styles are supported automatically:

    # Azure AI Foundry / newer resources (OpenAI-compatible v1 API):
    $env:AZURE_OPENAI_ENDPOINT = "https://<resource>.services.ai.azure.com/openai/v1"
    $env:AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-mini"

    # Classic Azure OpenAI resource:
    $env:AZURE_OPENAI_ENDPOINT = "https://<resource>.openai.azure.com"
    $env:AZURE_OPENAI_DEPLOYMENT = "<deployment name>"
    # optional: $env:AZURE_OPENAI_API_VERSION = "2024-06-01"

    # both need:
    $env:AZURE_OPENAI_API_KEY = "..."      # your own terminal, not via an AI tool

If these aren't set, `llm_available()` returns False and callers should fall
back to the keyword heuristic in `detect_vague_prompt` (app.cost_predictor).
Any network/parse failure here returns None -- never raises into the CLI.
"""
from __future__ import annotations

import json
import os

import requests

_DEFAULT_API_VERSION = "2024-06-01"
_TIMEOUT_SECONDS = 15

_SYSTEM_PROMPT = (
    "You review a single coding-assistant prompt BEFORE it is sent, for two things:\n"
    "1) Vagueness/incompleteness -- missing a target file/function, ambiguous "
    "scope, or generic filler like 'fix it'/'clean this up'.\n"
    "2) Task intent and complexity, to route it to an appropriately-capable model "
    "tier (economy < standard < premium). Use 'economy' for simple/low-risk work "
    "(docs, small edits, boilerplate, tests), 'standard' for typical feature/bugfix "
    "work, and 'premium' only for complex/high-stakes work (architecture, tricky "
    "algorithms, security, large multi-file refactors).\n"
    "Respond with ONLY compact JSON: {\"vague\": true|false, \"reasons\": [\"...\"], "
    "\"rewritten_prompt\": \"...\" or null, \"intent\": \"short label\", "
    "\"complexity\": \"low|medium|high\", \"recommended_tier\": "
    "\"economy|standard|premium\", \"routing_reason\": \"one short sentence\"}. "
    "If not vague, reasons=[] and rewritten_prompt=null."
)


def llm_available() -> bool:
    """True if Azure OpenAI credentials are set in the environment."""
    return bool(
        os.environ.get("AZURE_OPENAI_API_KEY")
        and os.environ.get("AZURE_OPENAI_ENDPOINT")
        and os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    )


def _build_request_target() -> tuple[str, dict]:
    """Return (url, headers) for the configured endpoint.

    Supports three surfaces:
    - Azure AI Foundry project endpoint (`https://<res>.services.ai.azure.com/api/projects/<proj>`)
      -> `{endpoint}/openai/v1/chat/completions`, `api-key` header.
    - OpenAI-compatible v1 API (endpoint already contains `/openai/v1`)
      -> `{base}/chat/completions`, Bearer auth.
    - Classic Azure OpenAI, where the endpoint is the resource base
      (`https://<res>.openai.azure.com`) -> deployments path + api-version,
      `api-key` header.
    """
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    key = os.environ["AZURE_OPENAI_API_KEY"]

    if "/api/projects/" in endpoint:
        url = f"{endpoint}/openai/v1/chat/completions"
        headers = {"api-key": key, "Content-Type": "application/json"}
        return url, headers

    if "/openai/v1" in endpoint:
        base = endpoint[: endpoint.index("/openai/v1") + len("/openai/v1")]
        url = f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        return url, headers

    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", _DEFAULT_API_VERSION)
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {"api-key": key, "Content-Type": "application/json"}
    return url, headers


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def review_prompt_with_llm(prompt: str) -> dict | None:
    """Ask Azure OpenAI whether `prompt` is vague, and a rewrite if so.

    Returns {"vague": bool, "reasons": list[str], "rewritten_prompt": str|None}
    or None if the LLM is not configured or the call fails for any reason.
    """
    if not llm_available():
        return None

    url, headers = _build_request_target()

    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": os.environ["AZURE_OPENAI_DEPLOYMENT"],
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 300,
            },
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return None

    parsed = _extract_json(content)
    if not isinstance(parsed, dict) or "vague" not in parsed:
        return None
    tier = parsed.get("recommended_tier")
    if tier not in ("economy", "standard", "premium"):
        tier = None
    return {
        "vague": bool(parsed.get("vague")),
        "reasons": list(parsed.get("reasons") or []),
        "rewritten_prompt": parsed.get("rewritten_prompt") or None,
        "intent": parsed.get("intent") or None,
        "complexity": parsed.get("complexity") or None,
        "recommended_tier": tier,
        "routing_reason": parsed.get("routing_reason") or None,
    }
