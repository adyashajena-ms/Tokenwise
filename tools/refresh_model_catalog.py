"""Regenerate the static model catalog from local VS Code model metadata.

Run this only when the available models change:

    python -m tools.refresh_model_catalog

It prints a MODEL_CATALOG dict you can paste into app/model_pricing.py. This is
the ONLY place that reads the logs for model metadata; runtime code uses the
baked-in catalog. The authoritative source is the Copilot models API (auth-gated),
which this script does not call.
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from app.chat_sessions import _find_session_files, _storage_root, reconstruct_session


def _collect(catalog: dict, node) -> None:
    if isinstance(node, dict):
        if "identifier" in node and "metadata" in node:
            ident = node.get("identifier")
            meta = node.get("metadata") or {}
            if ident and ident not in catalog:
                catalog[ident] = {
                    "context_window": meta.get("maxInputTokens"),
                    "max_output_tokens": meta.get("maxOutputTokens"),
                    "family": meta.get("family"),
                    "vendor": meta.get("vendor"),
                    "vision": (meta.get("capabilities") or {}).get("vision"),
                    "tool_calling": (meta.get("capabilities") or {}).get("toolCalling"),
                    "agent_mode": (meta.get("capabilities") or {}).get("agentMode"),
                }
        for value in node.values():
            _collect(catalog, value)
    elif isinstance(node, list):
        for item in node:
            _collect(catalog, item)


_COST_KEYS = {"inputCost", "outputCost", "cacheCost", "cacheWriteCost"}


def _collect_pricing(pricing: dict, node, hint=None) -> None:
    if isinstance(node, dict):
        mid = node.get("modelId") or node.get("identifier")
        if mid:
            hint = mid
        if _COST_KEYS & set(node.keys()) and hint and hint not in pricing:
            pricing[hint] = {
                "input": node.get("inputCost"),
                "output": node.get("outputCost"),
                "cache_read": node.get("cacheCost"),
                "cache_write": node.get("cacheWriteCost"),
            }
        for value in node.values():
            _collect_pricing(pricing, value, hint)
    elif isinstance(node, list):
        for item in node:
            _collect_pricing(pricing, item, hint)


def main() -> int:
    catalog: dict = {}
    pricing: dict = {}
    for _workspace, path in _find_session_files(_storage_root()):
        state = reconstruct_session(path)
        if isinstance(state, dict):
            _collect(catalog, state)
            _collect_pricing(pricing, state)

    print(f"# {len(catalog)} models found\nMODEL_CATALOG = {{")
    for model_id, details in catalog.items():
        print(f'    "{model_id}": {details},')
    print("}")

    print(f"\n# real credit pricing per 1M tokens ({len(pricing)} models)\nMODEL_PRICING = {{")
    for model_id, rates in pricing.items():
        print(f'    "{model_id}": {rates},')
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
