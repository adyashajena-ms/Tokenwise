import pytest

from app.estimator import estimate_presales_cost
from app.models import Attempt, Case, Engagement, get_session


def _seed_cases(session, task_type, n_cases, cost_low, cost_high, seed=1):
    import numpy as np
    rng = np.random.default_rng(seed)
    engagement = Engagement(customer="TestCo", task_type=task_type, complexity_tier="low")
    session.add(engagement)
    session.flush()
    for _ in range(n_cases):
        case = Case(engagement_id=engagement.id, status="accepted")
        session.add(case)
        session.flush()
        cost = float(rng.uniform(cost_low, cost_high))
        session.add(Attempt(case_id=case.id, attempt_number=1, tokens_used=1000, cost=cost, status="accepted"))


def test_percentiles_are_ordered():
    with get_session() as session:
        _seed_cases(session, "narrow", n_cases=50, cost_low=9.0, cost_high=11.0)

    result = estimate_presales_cost(task_type="narrow", volume=20, seed=123)
    assert result["p10"] <= result["p50"] <= result["p90"]


def test_wider_historical_variance_widens_interval():
    with get_session() as session:
        _seed_cases(session, "narrow", n_cases=50, cost_low=9.0, cost_high=11.0, seed=1)
        _seed_cases(session, "wide", n_cases=50, cost_low=1.0, cost_high=50.0, seed=2)

    narrow = estimate_presales_cost(task_type="narrow", volume=20, seed=123)
    wide = estimate_presales_cost(task_type="wide", volume=20, seed=123)

    assert (wide["p90"] - wide["p10"]) > (narrow["p90"] - narrow["p10"])


def test_low_sample_size_flagged_as_low_confidence():
    with get_session() as session:
        _seed_cases(session, "sparse", n_cases=5, cost_low=5.0, cost_high=6.0)

    result = estimate_presales_cost(task_type="sparse", volume=10, seed=123)
    assert "low" in result["assumptions"]["confidence"]


def test_unknown_task_type_raises():
    with get_session():
        pass
    with pytest.raises(ValueError):
        estimate_presales_cost(task_type="does_not_exist", volume=10)
