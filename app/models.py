"""SQLAlchemy data model for the AI Yield Ledger.

Unit of work hierarchy: Engagement -> Case -> Attempt.
A Case is one thing that needs a single accepted outcome (e.g. one ticket,
one migrated workload). It may take several Attempts (retries) before an
attempt is accepted, or the Case may be abandoned with no accepted outcome.
"""
from __future__ import annotations

import datetime
from contextlib import contextmanager

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

DB_PATH = "sqlite:///hack2026.db"


class Base(DeclarativeBase):
    pass


class Engagement(Base):
    __tablename__ = "engagements"

    id = Column(Integer, primary_key=True)
    customer = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    complexity_tier = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    cases = relationship("Case", back_populates="engagement", cascade="all, delete-orphan")


class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True)
    engagement_id = Column(Integer, ForeignKey("engagements.id"), nullable=False)
    # "accepted": one attempt was accepted. "abandoned": no attempt accepted, work stopped.
    status = Column(String, nullable=False)
    # How the accepted/abandoned status was decided: "synthetic", "git-diff",
    # or "transcript-heuristic". Lets the ledger show which outcomes are
    # git-verified vs. inferred.
    outcome_source = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    engagement = relationship("Engagement", back_populates="cases")
    attempts = relationship("Attempt", back_populates="case", cascade="all, delete-orphan")


class Attempt(Base):
    __tablename__ = "attempts"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    attempt_number = Column(Integer, nullable=False)
    tokens_used = Column(Integer, nullable=False)
    cost = Column(Float, nullable=False)
    # "accepted": this attempt's artifact was accepted. "rejected": retried after this.
    status = Column(String, nullable=False)
    rework_hours = Column(Float, default=0.0)
    rework_cost = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    case = relationship("Case", back_populates="attempts")


_engine = create_engine(DB_PATH, echo=False)
SessionLocal = sessionmaker(bind=_engine)


def init_db() -> None:
    Base.metadata.create_all(_engine)
    # Lightweight migration: add outcome_source to a pre-existing cases table
    # so older databases keep working without being wiped.
    existing = {col["name"] for col in inspect(_engine).get_columns("cases")}
    if "outcome_source" not in existing:
        with _engine.begin() as conn:
            conn.execute(text("ALTER TABLE cases ADD COLUMN outcome_source VARCHAR"))


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
