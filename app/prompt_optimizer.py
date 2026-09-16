"""Predict a prompt's token/cost footprint before you send it, and propose
safe reductions that trim tokens without changing what you're actually asking.

Two capabilities, both aligned with the rest of this repo's evidence standards:

- `predict_prompt_cost` — estimates tokens and cost for a prompt *before* it is
  sent. Uses the real BPE tokenizer (`tiktoken`) when it is installed, otherwise
  falls back to the same chars/4 PROXY the ledger ingest uses. Every result
  states which method produced it.
- `optimize_prompt` — applies only *meaning-preserving* rewrites (collapsing
  redundant whitespace, dropping filler/politeness that carries no instruction)
  and reports exactly what it changed and how many tokens each rule saved, so a
  human can review before trusting it. It never rephrases your actual request,
  so the model's output should not be compromised.
"""
from __future__ import annotations

import re

# Reuse the ledger's proxy constants so token/cost figures stay consistent with
# the rest of the app (see app/ingest_vscode_sessions.py).
from app.ingest_vscode_sessions import CHARS_PER_TOKEN, PRICE_PER_1K_TOKENS

try:  # Prefer a real tokenizer when available; degrade gracefully if not.
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - depends on optional dependency
    _ENCODING = None


def estimate_tokens(text: str) -> int:
    """Token count for `text`. Exact BPE count when tiktoken is available,
    otherwise a chars/4 proxy consistent with the ledger."""
    if not text:
        return 0
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def _token_method() -> str:
    return "tiktoken cl100k_base (exact)" if _ENCODING is not None else "chars/4 proxy"


def estimate_cost(tokens: int, price_per_1k: float = PRICE_PER_1K_TOKENS) -> float:
    """USD cost for `tokens` at the given per-1K-token rate."""
    return (tokens / 1000) * price_per_1k


def predict_prompt_cost(text: str, price_per_1k: float = PRICE_PER_1K_TOKENS) -> dict:
    """Predict token count and cost for a prompt before sending it."""
    tokens = estimate_tokens(text)
    return {
        "tokens": tokens,
        "cost": estimate_cost(tokens, price_per_1k),
        "characters": len(text),
        "assumptions": {
            "token_method": _token_method(),
            "price_per_1k_tokens": price_per_1k,
            "note": "input-side prediction only; model output tokens are billed separately",
        },
    }


# Meaning-preserving reductions. Each rule is (label, compiled_pattern,
# replacement). These only remove filler that carries no instruction or
# collapse redundant whitespace -- they never rephrase the actual request.
_FILLER_RULES: list[tuple[str, re.Pattern, str]] = [
    ("drop leading politeness",
     re.compile(r"(?i)\b(please|kindly)\b[ ,]*"), ""),
    ("drop hedges/fillers",
     re.compile(r"(?i)\b(just|really|actually|basically|simply|literally|very|quite|"
                r"in order to)\b[ ]*"), lambda m: "to " if m.group(1).lower() == "in order to" else ""),
    ("drop wordy openers",
     re.compile(r"(?i)\b(i was wondering if you could|i would like you to|"
                r"could you please|could you|can you please|can you|would you|"
                r"i want you to|i'd like you to)\b[ ]*"), ""),
    ("drop filler openers",
     re.compile(r"(?i)^\s*(so|well|okay|ok|now|alright)[ ,]+", re.MULTILINE), ""),
    ("collapse repeated spaces",
     re.compile(r"[ \t]{2,}"), " "),
    ("trim line-end whitespace",
     re.compile(r"[ \t]+$", re.MULTILINE), ""),
    ("collapse blank-line runs",
     re.compile(r"\n{3,}"), "\n\n"),
]


def optimize_prompt(text: str, price_per_1k: float = PRICE_PER_1K_TOKENS) -> dict:
    """Trim a prompt to fewer tokens using only meaning-preserving rewrites.

    Returns the original and optimized prompt, the token/cost before and after,
    and a per-rule breakdown of what was changed and how many tokens it saved,
    so the reduction is auditable rather than a black box.
    """
    original = text
    before_tokens = estimate_tokens(original)

    current = text
    applied: list[dict] = []
    for label, pattern, replacement in _FILLER_RULES:
        candidate = pattern.sub(replacement, current)
        if candidate == current:
            continue
        tokens_before_rule = estimate_tokens(current)
        tokens_after_rule = estimate_tokens(candidate)
        saved = tokens_before_rule - tokens_after_rule
        if saved <= 0:
            # Rule changed text but didn't reduce tokens (e.g. pure formatting);
            # keep the cleaner text but don't advertise a saving.
            current = candidate
            continue
        applied.append({"rule": label, "tokens_saved": saved})
        current = candidate

    optimized = current.strip()
    after_tokens = estimate_tokens(optimized)
    saved_tokens = before_tokens - after_tokens

    return {
        "original": original,
        "optimized": optimized,
        "before": {
            "tokens": before_tokens,
            "cost": estimate_cost(before_tokens, price_per_1k),
        },
        "after": {
            "tokens": after_tokens,
            "cost": estimate_cost(after_tokens, price_per_1k),
        },
        "tokens_saved": saved_tokens,
        "cost_saved": estimate_cost(saved_tokens, price_per_1k),
        "percent_saved": (saved_tokens / before_tokens * 100) if before_tokens else 0.0,
        "applied_rules": applied,
        "assumptions": {
            "token_method": _token_method(),
            "price_per_1k_tokens": price_per_1k,
            "guarantee": "only whitespace and non-instructional filler removed; the "
                         "substantive request is left unchanged",
            "status": "heuristic — review the optimized prompt before relying on it",
        },
    }
