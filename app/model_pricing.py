"""Real per-model pricing in CREDITS per 1,000,000 tokens.

These rates are the authoritative values shown in the VS Code Copilot model
picker (Credits per 1M tokens), extracted from local model metadata. Cost is
reported in credits, not dollars. Cache reads are much cheaper than fresh input;
cache writes cost more.

MODEL_CATALOG (context window, output cap, capabilities) and MODEL_PRICING were
extracted once from local VS Code model metadata and are baked in here so runtime
never reads the logs. The authoritative source is the Copilot models API
(auth-gated). Refresh with tools/refresh_model_catalog if the line-up changes.
"""
from __future__ import annotations

# Static per-model capabilities. Context window / output cap in tokens.
MODEL_CATALOG: dict[str, dict] = {
    "copilot/gpt-5.6-sol": {"context_window": 921793, "max_output_tokens": 128000,
                            "family": "gpt-5.6-sol", "vendor": "copilot",
                            "vision": True, "tool_calling": True, "agent_mode": True},
    "copilot/claude-opus-4.8": {"context_window": 935793, "max_output_tokens": 64000,
                                "family": "claude-opus-4.8", "vendor": "copilot",
                                "vision": True, "tool_calling": True, "agent_mode": True},
    "copilot/claude-sonnet-5": {"context_window": 935793, "max_output_tokens": 64000,
                                "family": "claude-sonnet-5", "vendor": "copilot",
                                "vision": True, "tool_calling": True, "agent_mode": True},
}

# CREDITS per 1M tokens (input, output, cache_read, cache_write). Some models
# also have a `long_context` tier that applies above a large-input threshold.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "copilot/gpt-5.6-sol": {"input": 400, "output": 2000, "cache_read": 40, "cache_write": 500,
                            "long_context": {"input": 800, "output": 3000, "cache_read": 80, "cache_write": 1000}},
    "copilot/gpt-5.6-terra": {"input": 200, "output": 1200, "cache_read": 20, "cache_write": 250,
                              "long_context": {"input": 400, "output": 1800, "cache_read": 40, "cache_write": 500}},
    "copilot/gpt-5.4": {"input": 250, "output": 1500, "cache_read": 25, "cache_write": 0,
                        "long_context": {"input": 500, "output": 2250, "cache_read": 50, "cache_write": 0}},
    "copilot/claude-opus-4.8": {"input": 500, "output": 2500, "cache_read": 50, "cache_write": 625},
    "copilot/claude-sonnet-5": {"input": 200, "output": 1000, "cache_read": 20, "cache_write": 250},
    "copilot/claude-haiku-4.5": {"input": 100, "output": 500, "cache_read": 10, "cache_write": 125},
}

# Credits per 1M tokens for an unknown model (mid-tier default).
DEFAULT_PRICING = {"input": 500, "output": 2500, "cache_read": 50, "cache_write": 625}

_PER_MILLION = 1_000_000


def pricing_for(model_id: str | None) -> dict[str, float]:
    """Return the credit rates (per 1M tokens) for a model, or a default if unknown."""
    if model_id and model_id in MODEL_PRICING:
        return MODEL_PRICING[model_id]
    return DEFAULT_PRICING


def model_details(model_id: str | None) -> dict | None:
    """Static catalog entry (context window, capabilities) for a model."""
    if model_id and model_id in MODEL_CATALOG:
        return MODEL_CATALOG[model_id]
    return None


def long_context_pricing_for(model_id: str | None) -> dict[str, float] | None:
    """Long-context tier rates for a model, if it has one."""
    price = pricing_for(model_id)
    lc = price.get("long_context")
    return lc if isinstance(lc, dict) else None


def request_cost(prompt_tokens: int, output_tokens: int, model_id: str | None) -> float:
    """Credits for one request given token counts and a model."""
    price = pricing_for(model_id)
    return (prompt_tokens / _PER_MILLION) * price["input"] + (output_tokens / _PER_MILLION) * price["output"]


def request_cost_detailed(
    prompt_tokens: int,
    output_tokens: int,
    model_id: str | None,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Credits accounting for cache reads (cheaper), cache writes, and reasoning tokens."""
    price = pricing_for(model_id)
    cached = min(max(cached_tokens, 0), prompt_tokens)
    fresh = prompt_tokens - cached
    input_credits = (
        (fresh / _PER_MILLION) * price["input"]
        + (cached / _PER_MILLION) * price["cache_read"]
        + (max(cache_write_tokens, 0) / _PER_MILLION) * price["cache_write"]
    )
    output_credits = ((output_tokens + max(reasoning_tokens, 0)) / _PER_MILLION) * price["output"]
    return input_credits + output_credits


def cheapest_model(prompt_tokens: int, output_tokens: int) -> tuple[str, float]:
    """Cheapest known model (in credits) for a given token footprint, with its cost."""
    best_model = min(
        MODEL_PRICING,
        key=lambda m: request_cost(prompt_tokens, output_tokens, m),
    )
    return best_model, request_cost(prompt_tokens, output_tokens, best_model)


# Rough capability tiers for intent-based routing (economy < standard < premium).
# Within a tier, the cheapest known model is chosen. Used to balance quality vs.
# cost instead of always defaulting to the single cheapest model.
MODEL_TIERS: dict[str, list[str]] = {
    "economy": ["copilot/claude-haiku-4.5", "copilot/claude-sonnet-5", "copilot/gpt-5.6-terra"],
    "standard": ["copilot/gpt-5.4", "copilot/gpt-5.6-sol"],
    "premium": ["copilot/claude-opus-4.8"],
}


def recommend_model(tier: str, prompt_tokens: int, output_tokens: int) -> tuple[str, float]:
    """Cheapest known model within the given tier; falls back to overall cheapest."""
    known = [m for m in MODEL_TIERS.get(tier, []) if m in MODEL_PRICING]
    if not known:
        return cheapest_model(prompt_tokens, output_tokens)
    best_model = min(known, key=lambda m: request_cost(prompt_tokens, output_tokens, m))
    return best_model, request_cost(prompt_tokens, output_tokens, best_model)
