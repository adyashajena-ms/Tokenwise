"""Convert Copilot credits to real-world currency.

Copilot does not publish an exact credits->USD exchange rate, so the rate here
is CONFIGURABLE, not authoritative. Override it with --credits-per-dollar on the
CLI (or the credits_per_dollar argument) once you know your actual plan's rate.
"""
from __future__ import annotations

# Default assumption: $10 of plan spend ~= 1,000 credits. Override if you know
# your actual rate (e.g. from your Copilot billing page).
DEFAULT_CREDITS_PER_DOLLAR = 100.0

# USD -> INR. Override with --usd-to-inr; markets move, this is a snapshot default.
DEFAULT_USD_TO_INR = 83.0


def credits_to_usd(credits: float, credits_per_dollar: float = DEFAULT_CREDITS_PER_DOLLAR) -> float:
    if credits_per_dollar <= 0:
        raise ValueError("credits_per_dollar must be positive")
    return credits / credits_per_dollar


def credits_to_inr(
    credits: float,
    credits_per_dollar: float = DEFAULT_CREDITS_PER_DOLLAR,
    usd_to_inr: float = DEFAULT_USD_TO_INR,
) -> float:
    if usd_to_inr <= 0:
        raise ValueError("usd_to_inr must be positive")
    return credits_to_usd(credits, credits_per_dollar) * usd_to_inr
