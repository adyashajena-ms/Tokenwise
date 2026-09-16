"""Presales cost estimator.

Produces a P10/P50/P90 cost interval for a prospective engagement via Monte
Carlo bootstrap resampling of historical accepted-outcome costs. This is a
PROPOSED estimation method, not a validated statistical model — see the
`assumptions` block returned with every estimate for its stated limits.
"""
from __future__ import annotations

import numpy as np

from app.ledger import accepted_case_costs

# Rough, stated (not calibrated) adjustment for complexity tiers that differ
# from the historical task type's own tier.
COMPLEXITY_MULTIPLIER = {"low": 0.8, "medium": 1.0, "high": 1.4}
MIN_SAMPLE_FOR_MODERATE_CONFIDENCE = 30


def estimate_presales_cost(
    task_type: str,
    volume: int,
    complexity_tier: str | None = None,
    n_simulations: int = 5000,
    seed: int | None = None,
) -> dict:
    """Estimate total cost for `volume` cases of `task_type` via bootstrap Monte Carlo.

    Returns P10/P50/P90 of total cost plus an `assumptions` block describing
    exactly how the estimate was produced, per the evidence-standards ask.
    """
    historical_costs = accepted_case_costs(task_type=task_type)
    sample_size = len(historical_costs)

    if sample_size == 0:
        raise ValueError(f"No historical accepted-outcome data for task_type={task_type!r}")

    rng = np.random.default_rng(seed)
    values = historical_costs.to_numpy()

    multiplier = 1.0
    if complexity_tier and complexity_tier in COMPLEXITY_MULTIPLIER:
        multiplier = COMPLEXITY_MULTIPLIER[complexity_tier]

    # Bootstrap: draw `volume` historical per-case costs (with replacement) per
    # simulated engagement, repeat n_simulations times, sum each simulation.
    draws = rng.choice(values, size=(n_simulations, volume), replace=True) * multiplier
    totals = draws.sum(axis=1)

    p10, p50, p90 = np.percentile(totals, [10, 50, 90])

    confidence_note = (
        "moderate: based on >=30 historical accepted outcomes"
        if sample_size >= MIN_SAMPLE_FOR_MODERATE_CONFIDENCE
        else f"low: only {sample_size} historical accepted outcomes available"
    )

    return {
        "task_type": task_type,
        "volume": volume,
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "assumptions": {
            "method": "empirical bootstrap Monte Carlo over historical accepted-outcome costs",
            "historical_sample_size": int(sample_size),
            "n_simulations": n_simulations,
            "complexity_adjustment": f"{complexity_tier or 'none'} (x{multiplier})",
            "confidence": confidence_note,
            "status": "proposed method — not independently validated against held-out engagements",
        },
    }
