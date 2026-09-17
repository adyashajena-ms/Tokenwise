import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.models as models_module
from app.models import Base


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point the models module at a throwaway sqlite file for each test."""
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}")
    monkeypatch.setattr(models_module, "_engine", engine)
    monkeypatch.setattr(models_module, "SessionLocal", sessionmaker(bind=engine))
    Base.metadata.create_all(engine)
    yield


@pytest.fixture(autouse=True)
def _no_real_llm_calls(monkeypatch):
    """Clear Azure creds by default so tests never hit the real network.

    Tests that exercise the LLM path set these explicitly via monkeypatch.
    """
    for var in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"):
        monkeypatch.delenv(var, raising=False)
    yield
