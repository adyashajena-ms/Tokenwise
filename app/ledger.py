"""Yield ledger: turns raw spend events into cost-per-accepted-outcome metrics.

Every number here is traceable back to the raw Attempt rows it was built
from (see trace_case) rather than being a black-box aggregate.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from app.models import Attempt, Case, Engagement, get_session


def _raw_frame(session: Session) -> pd.DataFrame:
    rows = (
        session.query(
            Case.id.label("case_id"),
            Case.status.label("case_status"),
            Case.outcome_source.label("outcome_source"),
            Engagement.id.label("engagement_id"),
            Engagement.customer.label("customer"),
            Engagement.task_type.label("task_type"),
            Engagement.complexity_tier.label("complexity_tier"),
            Attempt.id.label("attempt_id"),
            Attempt.attempt_number.label("attempt_number"),
            Attempt.status.label("attempt_status"),
            Attempt.cost.label("cost"),
            Attempt.rework_cost.label("rework_cost"),
            Attempt.tokens_used.label("tokens_used"),
        )
        .join(Engagement, Case.engagement_id == Engagement.id)
        .join(Attempt, Attempt.case_id == Case.id)
        .all()
    )
    return pd.DataFrame(rows, columns=[
        "case_id", "case_status", "outcome_source", "engagement_id", "customer", "task_type",
        "complexity_tier", "attempt_id", "attempt_number", "attempt_status",
        "cost", "rework_cost", "tokens_used",
    ])


def yield_ledger_by_task_type() -> pd.DataFrame:
    """One row per task_type with cost-per-accepted-outcome and waste breakdown."""
    with get_session() as session:
        df = _raw_frame(session)

    if df.empty:
        return pd.DataFrame()

    df["attempt_total_cost"] = df["cost"] + df["rework_cost"]

    records = []
    for task_type, group in df.groupby("task_type"):
        cases = group.groupby("case_id").agg(
            case_status=("case_status", "first"),
            outcome_source=("outcome_source", "first"),
        ).reset_index()
        accepted_case_ids = set(cases.loc[cases.case_status == "accepted", "case_id"])
        abandoned_case_ids = set(cases.loc[cases.case_status == "abandoned", "case_id"])
        git_verified_cases = int((cases.outcome_source == "git-diff").sum())

        total_spend = group["attempt_total_cost"].sum()
        accepted_mask = group["case_id"].isin(accepted_case_ids)
        abandoned_mask = group["case_id"].isin(abandoned_case_ids)

        # Spend attributed to accepted outcomes = everything spent (incl. retries/rework)
        # on cases that ended in an accepted attempt.
        spend_on_accepted_cases = group.loc[accepted_mask, "attempt_total_cost"].sum()
        winning_attempt_cost = group.loc[accepted_mask & (group["attempt_status"] == "accepted"), "cost"].sum()
        retry_waste = group.loc[accepted_mask & (group["attempt_status"] == "rejected"), "cost"].sum()
        rework_waste = group.loc[accepted_mask, "rework_cost"].sum()
        abandoned_waste = group.loc[abandoned_mask, "attempt_total_cost"].sum()

        n_accepted = len(accepted_case_ids)
        n_abandoned = len(abandoned_case_ids)

        records.append({
            "task_type": task_type,
            "customer": group["customer"].iloc[0],
            "complexity_tier": group["complexity_tier"].iloc[0],
            "total_spend": total_spend,
            "accepted_outcomes": n_accepted,
            "abandoned_cases": n_abandoned,
            "cost_per_accepted_outcome": (spend_on_accepted_cases / n_accepted) if n_accepted else None,
            "yield_ratio": (winning_attempt_cost / total_spend) if total_spend else None,
            "retry_waste": retry_waste,
            "rework_waste": rework_waste,
            "abandoned_waste": abandoned_waste,
            "git_verified_outcomes": git_verified_cases,
        })

    return pd.DataFrame(records).sort_values("task_type").reset_index(drop=True)


def trace_case(case_id: int) -> pd.DataFrame:
    """Full attempt-by-attempt spend history for one case, for drill-down/audit."""
    with get_session() as session:
        df = _raw_frame(session)
    return df.loc[df["case_id"] == case_id].sort_values("attempt_number").reset_index(drop=True)


def accepted_case_costs(task_type: str | None = None) -> pd.Series:
    """Total attributed cost per accepted outcome, one value per accepted case.

    Used as the empirical historical distribution feeding the presales estimator.
    """
    with get_session() as session:
        df = _raw_frame(session)
    if task_type:
        df = df.loc[df["task_type"] == task_type]
    df["attempt_total_cost"] = df["cost"] + df["rework_cost"]
    accepted_ids = df.loc[df["case_status"] == "accepted", "case_id"].unique()
    per_case = (
        df.loc[df["case_id"].isin(accepted_ids)]
        .groupby("case_id")["attempt_total_cost"]
        .sum()
    )
    return per_case
