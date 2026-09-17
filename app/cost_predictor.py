"""Predict a prompt's dollar cost across models BEFORE you send it.

Counts real input tokens for your prompt plus any attached context files (exact
BPE via tiktoken when available), predicts output tokens, and prices the request
against every known model so you can pick the cheapest that fits.

Costs use illustrative per-model rates (app.model_pricing) -- directional, not
billed figures.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from app.model_pricing import MODEL_PRICING, model_details, pricing_for
from app.prompt_optimizer import estimate_tokens

DEFAULT_OUTPUT_TOKENS = 500

# Only treat these as code-context files when auto-collecting from git.
_CODE_SUFFIXES = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".cs", ".java", ".go", ".rb", ".rs",
    ".cpp", ".c", ".h", ".hpp", ".xaml", ".json", ".yaml", ".yml", ".md", ".sql",
}
_MAX_AUTO_FILES = 12


def gather_git_context_files(repo_dir: str = ".") -> list[str]:
    """Auto-collect changed/untracked code files from git as prompt context."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "status", "--porcelain", "--untracked-files=all"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []

    files: list[str] = []
    for line in out.stdout.splitlines():
        # porcelain: 'XY <path>' (path starts at column 3)
        path = line[3:].strip().strip('"')
        if " -> " in path:  # renames
            path = path.split(" -> ", 1)[1]
        if not path:
            continue
        if Path(path).suffix.lower() in _CODE_SUFFIXES and Path(path).is_file():
            files.append(path)
        if len(files) >= _MAX_AUTO_FILES:
            break
    return files


def _read_context_files(paths: list[str] | None) -> tuple[int, list[dict]]:
    """Return total context tokens and a per-file token breakdown."""
    breakdown: list[dict] = []
    total = 0
    for raw in paths or []:
        path = Path(raw)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            breakdown.append({"file": raw, "tokens": 0, "error": str(exc)})
            continue
        tokens = estimate_tokens(text)
        total += tokens
        breakdown.append({"file": str(path), "tokens": tokens})
    return total, breakdown


def predict_across_models(
    prompt: str,
    context_files: list[str] | None = None,
    expected_output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    extra_context_tokens: int = 0,
) -> dict:
    """Predict input/output cost of a prompt for every known model."""
    if expected_output_tokens < 0:
        raise ValueError("expected_output_tokens must be non-negative")

    prompt_tokens = estimate_tokens(prompt)
    context_tokens, file_breakdown = _read_context_files(context_files)
    context_tokens += max(extra_context_tokens, 0)
    input_tokens = prompt_tokens + context_tokens

    models = []
    for model_id in MODEL_PRICING:
        price = pricing_for(model_id)
        details = model_details(model_id) or {}
        context_window = details.get("context_window")
        input_cost = (input_tokens / 1_000_000) * price["input"]
        output_cost = (expected_output_tokens / 1_000_000) * price["output"]
        models.append({
            "model": model_id,
            "context_window": context_window,
            "fits": (context_window is None) or (input_tokens <= context_window),
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        })
    models.sort(key=lambda m: m["total_cost"])

    return {
        "prompt_tokens": prompt_tokens,
        "context_tokens": context_tokens,
        "input_tokens": input_tokens,
        "expected_output_tokens": expected_output_tokens,
        "file_breakdown": file_breakdown,
        "models": models,
        "cheapest": models[0] if models else None,
        "most_expensive": models[-1] if models else None,
    }


def historical_median_output_tokens() -> int | None:
    """Median real output tokens from local chatSessions history, if available."""
    try:
        from app.prompt_analysis import requests_dataframe

        df = requests_dataframe()
        if df.empty:
            return None
        return int(df["output_tokens"].median())
    except Exception:
        return None


def historical_median_input_tokens() -> int | None:
    """Median real prompt (input) tokens per request from local history.

    This reflects the TRUE per-request context — referenced files, conversation
    history, and tool outputs — not just the files being edited, so it is a far
    more realistic input size than summing git-changed files.
    """
    try:
        from app.prompt_analysis import requests_dataframe

        df = requests_dataframe()
        real = df[df["prompt_tokens"] > 0]["prompt_tokens"] if not df.empty else None
        if real is None or real.empty:
            return None
        return int(real.median())
    except Exception:
        return None
