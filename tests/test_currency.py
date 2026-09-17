import pytest

from app.currency import credits_to_inr, credits_to_usd
from app.predict import main


def test_credits_to_usd_uses_rate():
    assert credits_to_usd(100, credits_per_dollar=100) == 1.0
    assert credits_to_usd(50, credits_per_dollar=50) == 1.0


def test_credits_to_inr_chains_usd_rate():
    usd = credits_to_usd(100, credits_per_dollar=100)
    assert credits_to_inr(100, credits_per_dollar=100, usd_to_inr=80) == usd * 80


def test_invalid_rates_rejected():
    with pytest.raises(ValueError):
        credits_to_usd(10, credits_per_dollar=0)
    with pytest.raises(ValueError):
        credits_to_inr(10, usd_to_inr=-1)


def test_cli_shows_usd_and_inr_columns(capsys):
    exit_code = main(["Write a unit test", "--output-tokens", "400", "--usd", "--inr"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "TOTAL $" in out
    assert "TOTAL Rs" in out
    assert "Currency rates:" in out


def test_cli_respects_custom_rates(capsys):
    exit_code = main([
        "Write a unit test", "--output-tokens", "400", "--usd",
        "--credits-per-dollar", "50",
    ])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "50 credits/$" in out
