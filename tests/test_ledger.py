import pytest

from app.ledger import accepted_case_costs, yield_ledger_by_task_type
from app.models import Attempt, Case, Engagement, get_session


def _seed_simple_case(session, task_type="unit_test_type", accepted=True,
                       attempt_costs=(10.0, 5.0), rework_cost=0.0):
    engagement = Engagement(customer="TestCo", task_type=task_type, complexity_tier="low")
    session.add(engagement)
    session.flush()
    case = Case(engagement_id=engagement.id, status="accepted" if accepted else "abandoned")
    session.add(case)
    session.flush()
    n = len(attempt_costs)
    for i, cost in enumerate(attempt_costs, start=1):
        is_last = i == n
        status = "accepted" if (is_last and accepted) else "rejected"
        session.add(Attempt(
            case_id=case.id, attempt_number=i, tokens_used=int(cost * 1000),
            cost=cost, status=status,
            rework_cost=rework_cost if (is_last and accepted) else 0.0,
        ))
    return case.id


def test_cost_per_accepted_outcome_known_inputs():
    with get_session() as session:
        # One accepted case: attempt1 cost=10 (rejected), attempt2 cost=5 (accepted), rework=2
        _seed_simple_case(session, attempt_costs=(10.0, 5.0), rework_cost=2.0)

    df = yield_ledger_by_task_type()
    row = df.iloc[0]
    assert row["accepted_outcomes"] == 1
    assert row["cost_per_accepted_outcome"] == pytest.approx(17.0)  # 10 + 5 + 2
    assert row["retry_waste"] == pytest.approx(10.0)
    assert row["rework_waste"] == pytest.approx(2.0)
    assert row["abandoned_waste"] == pytest.approx(0.0)


def test_abandoned_case_is_pure_waste_not_attributed():
    with get_session() as session:
        _seed_simple_case(session, attempt_costs=(3.0, 4.0), accepted=False)

    df = yield_ledger_by_task_type()
    row = df.iloc[0]
    assert row["accepted_outcomes"] == 0
    assert row["cost_per_accepted_outcome"] is None
    assert row["abandoned_waste"] == pytest.approx(7.0)


def test_accepted_case_costs_matches_ledger_total():
    with get_session() as session:
        _seed_simple_case(session, attempt_costs=(10.0, 5.0), rework_cost=2.0)
        _seed_simple_case(session, attempt_costs=(6.0,), rework_cost=0.0)

    per_case = accepted_case_costs(task_type="unit_test_type")
    assert sorted(per_case.tolist()) == pytest.approx(sorted([17.0, 6.0]))
