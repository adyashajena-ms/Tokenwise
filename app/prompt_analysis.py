"""Post-dev analysis over REAL chatSessions token usage.

Turns per-request token records (from app.chat_sessions) into:
- per-project (workspace folder) token + cost totals,
- a good/bad classification of the user's own prompts so they can learn,
- a pre-development cost prediction from their real historical usage.

Cost uses illustrative model prices (see app.model_pricing) -- directional, not
billed figures.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from app.chat_sessions import ChatRequestRecord, load_chat_requests
from app.model_pricing import cheapest_model, request_cost, request_cost_detailed


def requests_dataframe(records: list[ChatRequestRecord] | None = None) -> pd.DataFrame:
    """Per-request DataFrame with real tokens and computed cost."""
    records = records if records is not None else load_chat_requests()
    rows = []
    for r in records:
        cost = request_cost_detailed(
            r.prompt_tokens, r.output_tokens, r.model_id,
            cached_tokens=r.cached_tokens, reasoning_tokens=r.reasoning_tokens,
        )
        rows.append({
            "workspace": r.workspace,
            "session_id": r.session_id,
            "session_title": r.session_title,
            "request_id": r.request_id,
            "timestamp": r.timestamp,
            "model_id": r.model_id,
            "prompt_tokens": r.prompt_tokens,
            "output_tokens": r.output_tokens,
            "total_tokens": r.prompt_tokens + r.output_tokens,
            "cached_tokens": r.cached_tokens,
            "reasoning_tokens": r.reasoning_tokens,
            "context_length": r.context_length,
            "prompt_chars": r.prompt_chars,
            "prompt_text": r.prompt_text,
            "kept_edits": r.kept_edits,
            "elapsed_ms": r.elapsed_ms,
            "edited_files": list(r.edited_files),
            "cost": cost,
        })
    return pd.DataFrame(rows)


def tokens_by_project(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate real token usage and cost per project folder."""
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby("workspace")
        .agg(
            requests=("request_id", "count"),
            prompt_tokens=("prompt_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            cost=("cost", "sum"),
        )
        .reset_index()
        .sort_values("total_tokens", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def score_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """Classify the user's actual prompts as good / average / needs work.

    Only rows with prompt text are scored (tool/continuation turns are skipped).
    Heuristic signals, each stated so the user can learn:
    - wasted:   the request produced no output tokens,
    - bloated:  prompt tokens in the top quartile (too much context),
    - efficient: healthy output-per-prompt-token ratio.
    """
    if df.empty:
        return pd.DataFrame()

    prompts = df[df["prompt_chars"] > 0].copy()
    if prompts.empty:
        return pd.DataFrame()

    bloat_threshold = prompts["prompt_tokens"].quantile(0.75)
    prompts["efficiency"] = prompts["output_tokens"] / prompts["prompt_tokens"].clip(lower=1)
    median_efficiency = prompts["efficiency"].median()

    def classify(row) -> tuple[str, str]:
        reasons = []
        if row["output_tokens"] == 0:
            reasons.append("no output produced")
        if row["prompt_tokens"] >= bloat_threshold:
            reasons.append("large prompt/context")
        if row["efficiency"] >= median_efficiency and row["output_tokens"] > 0:
            reasons.append("good output-per-token")

        if row["output_tokens"] == 0:
            label = "needs work"
        elif row["prompt_tokens"] >= bloat_threshold and row["efficiency"] < median_efficiency:
            label = "needs work"
        elif row["efficiency"] >= median_efficiency:
            label = "good"
        else:
            label = "average"
        return label, ", ".join(reasons) if reasons else "ordinary"

    labels_reasons = prompts.apply(classify, axis=1, result_type="expand")
    prompts["quality"] = labels_reasons[0]
    prompts["why"] = labels_reasons[1]
    return prompts.sort_values("cost", ascending=False).reset_index(drop=True)


def cost_by_feature(df: pd.DataFrame, workspace: str | None = None) -> pd.DataFrame:
    """Cost per feature, where each chat session is treated as one feature.

    Sessions are labeled by their VS Code title (customTitle); untitled sessions
    fall back to their first prompt or session id.
    """
    if df.empty:
        return pd.DataFrame()
    scope = df if not workspace else df[df["workspace"] == workspace]
    if scope.empty:
        return pd.DataFrame()

    def _first_prompt(series: pd.Series) -> str:
        for text in series:
            if isinstance(text, str) and text.strip():
                return text.strip()[:60]
        return ""

    def _top_files(series: pd.Series, limit: int = 3) -> str:
        counter: Counter[str] = Counter()
        for files in series:
            if isinstance(files, list):
                counter.update(files)
        return ", ".join(name for name, _ in counter.most_common(limit))

    grouped = (
        scope.groupby("session_id")
        .agg(
            session_title=("session_title", "first"),
            workspace=("workspace", "first"),
            first_prompt=("prompt_text", _first_prompt),
            top_files=("edited_files", _top_files),
            requests=("request_id", "count"),
            prompt_tokens=("prompt_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            cost=("cost", "sum"),
        )
        .reset_index()
    )
    grouped["feature"] = grouped.apply(
        lambda r: r["session_title"] or r["first_prompt"] or r["session_id"][:8],
        axis=1,
    )
    ordered = grouped[[
        "feature", "top_files", "workspace", "requests", "prompt_tokens",
        "output_tokens", "total_tokens", "cost",
    ]]
    return ordered.sort_values("cost", ascending=False).reset_index(drop=True)


def savings_if_cheapest_model(df: pd.DataFrame) -> pd.DataFrame:
    """For each request, the cheapest model and the cost saved versus the one used."""
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    cheapest = out.apply(
        lambda r: cheapest_model(r["prompt_tokens"], r["output_tokens"]),
        axis=1,
        result_type="expand",
    )
    out["cheapest_model"] = cheapest[0]
    out["cheapest_cost"] = cheapest[1]
    out["potential_saving"] = (out["cost"] - out["cheapest_cost"]).clip(lower=0)
    return out


def predict_task_cost(
    df: pd.DataFrame,
    workspace: str | None = None,
    planned_requests: int = 20,
) -> dict:
    """Pre-dev cost prediction from real historical per-request tokens.

    Uses the median real prompt/output tokens (optionally scoped to a project)
    to project the cost of a planned coding task before any tokens are spent.
    """
    if df.empty:
        raise ValueError("No chat history available to predict from.")
    scope = df if not workspace else df[df["workspace"] == workspace]
    if scope.empty:
        raise ValueError(f"No history for workspace={workspace!r}")

    median_prompt = int(scope["prompt_tokens"].median())
    median_output = int(scope["output_tokens"].median())
    median_cost = float(scope["cost"].median())
    cheapest = cheapest_model(median_prompt, median_output)

    return {
        "workspace": workspace or "all projects",
        "planned_requests": planned_requests,
        "median_prompt_tokens": median_prompt,
        "median_output_tokens": median_output,
        "median_cost_per_request": median_cost,
        "predicted_total_cost": median_cost * planned_requests,
        "cheapest_model": cheapest[0],
        "cheapest_total_cost": cheapest[1] * planned_requests,
        "assumptions": {
            "basis": "median of real historical per-request tokens",
            "sample_size": int(len(scope)),
            "pricing": "illustrative per-model rates (app.model_pricing)",
            "status": "directional estimate, not a billed quote",
        },
    }
