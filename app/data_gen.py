"""Synthetic spend/outcome data generator.

Seeds three task-type archetypes with deliberately different cost/retry/
abandonment profiles so the yield ledger and estimator have realistic
variance to show (this data is synthetic, not observed production data).
"""
from __future__ import annotations

import numpy as np

from app.models import Attempt, Case, Engagement, get_session, init_db

HOURLY_REWORK_RATE = 75.0  # $/hour for human rework, used to cost rework_hours
MAX_ATTEMPTS_PER_CASE = 6

ARCHETYPES = {
    "stable_low_variance": {
        "complexity_tier": "low",
        "customer": "Contoso",
        "cost_lognormal_mu": np.log(0.40),
        "cost_lognormal_sigma": 0.25,
        "spike_prob": 0.0,
        "spike_multiplier": 1.0,
        "retry_lambda": 0.3,
        "abandon_rate": 0.02,
        "rework_prob": 0.05,
        "rework_hours_mean": 0.2,
    },
    "retry_prone": {
        "complexity_tier": "medium",
        "customer": "Fabrikam",
        "cost_lognormal_mu": np.log(0.60),
        "cost_lognormal_sigma": 0.45,
        "spike_prob": 0.02,
        "spike_multiplier": 6.0,
        "retry_lambda": 2.0,
        "abandon_rate": 0.10,
        "rework_prob": 0.35,
        "rework_hours_mean": 0.75,
    },
    "rare_catastrophic_outlier": {
        "complexity_tier": "high",
        "customer": "Northwind",
        "cost_lognormal_mu": np.log(0.55),
        "cost_lognormal_sigma": 0.5,
        "spike_prob": 0.06,
        "spike_multiplier": 25.0,
        "retry_lambda": 0.6,
        "abandon_rate": 0.15,
        "rework_prob": 0.20,
        "rework_hours_mean": 0.5,
    },
}

TOKENS_PER_DOLLAR = 50_000  # rough conversion for a plausible tokens_used figure


def _sample_attempt_cost(rng: np.random.Generator, params: dict) -> float:
    cost = rng.lognormal(mean=params["cost_lognormal_mu"], sigma=params["cost_lognormal_sigma"])
    if rng.random() < params["spike_prob"]:
        cost *= params["spike_multiplier"]
    return float(cost)


def generate_dataset(cases_per_archetype: int = 150, seed: int = 7) -> None:
    """Populate the database with synthetic engagements/cases/attempts."""
    init_db()
    rng = np.random.default_rng(seed)

    with get_session() as session:
        for task_type, params in ARCHETYPES.items():
            engagement = Engagement(
                customer=params["customer"],
                task_type=task_type,
                complexity_tier=params["complexity_tier"],
            )
            session.add(engagement)
            session.flush()

            for _ in range(cases_per_archetype):
                n_attempts = min(int(rng.poisson(params["retry_lambda"])) + 1, MAX_ATTEMPTS_PER_CASE)
                abandoned = rng.random() < params["abandon_rate"]
                case = Case(
                    engagement_id=engagement.id,
                    status="abandoned" if abandoned else "accepted",
                    outcome_source="synthetic",
                )
                session.add(case)
                session.flush()

                for attempt_number in range(1, n_attempts + 1):
                    is_final_attempt = attempt_number == n_attempts
                    cost = _sample_attempt_cost(rng, params)
                    tokens_used = max(1, int(cost * TOKENS_PER_DOLLAR))

                    rework_hours = 0.0
                    rework_cost = 0.0
                    if is_final_attempt and not abandoned:
                        status = "accepted"
                        if rng.random() < params["rework_prob"]:
                            rework_hours = float(rng.exponential(params["rework_hours_mean"]))
                            rework_cost = rework_hours * HOURLY_REWORK_RATE
                    else:
                        status = "rejected"

                    session.add(
                        Attempt(
                            case_id=case.id,
                            attempt_number=attempt_number,
                            tokens_used=tokens_used,
                            cost=cost,
                            status=status,
                            rework_hours=rework_hours,
                            rework_cost=rework_cost,
                        )
                    )


if __name__ == "__main__":
    generate_dataset()
    print("Synthetic dataset generated at hack2026.db")
