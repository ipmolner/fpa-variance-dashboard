"""
test_commentary.py
------------------
Tests for commentary.py. Run with:  pytest -q

Each test pins one classification rule or one piece of phrasing, so a
future change that breaks the logic fails loudly rather than silently
producing plausible-sounding but wrong commentary.
"""

import pandas as pd
import pytest

from commentary import (
    CommentaryConfig,
    Pattern,
    add_commentary,
    classify_pattern,
    driver_phrase,
    fmt_money,
    fmt_month,
    generate_commentary,
)
from variance_engine import build_variance_report, flag_variances, load_data


def _history(pairs):
    """pairs: list of (month, variance_pct)."""
    return pd.DataFrame(pairs, columns=["month", "variance_pct"])


# --- formatting ------------------------------------------------------------
def test_money_formatting_scales():
    assert fmt_money(950) == "$950"
    assert fmt_money(75_968) == "$76K"
    assert fmt_money(-1_240_000) == "$1.2M"


def test_money_formatting_signed_option():
    # default drops the sign (commentary states direction in words)
    assert fmt_money(-75_968) == "$76K"
    # signed=True keeps it (tables and KPI cards need it)
    assert fmt_money(-75_968, signed=True) == "-$76K"
    assert fmt_money(75_968, signed=True) == "$76K"


def test_month_formatting():
    assert fmt_month("2024-12") == "Dec 2024"


def test_driver_phrase_flips_with_direction():
    assert "higher media" in driver_phrase("Advertising & Media", True)
    assert "reduced media" in driver_phrase("Advertising & Media", False)


def test_unknown_account_falls_back_gracefully():
    assert driver_phrase("Some New Account", True) == (
        "higher than planned spend in this account"
    )


# --- seasonal rule ---------------------------------------------------------
def test_seasonal_fires_on_comparable_prior_year_breach():
    hist = _history([("2024-12", 0.26), ("2025-06", 0.01), ("2025-12", 0.22)])
    pattern, ev = classify_pattern("2025-12", hist)
    assert pattern is Pattern.SEASONAL
    assert ev["quarter"] == "Q4"


def test_seasonal_rejects_much_smaller_prior_year_breach():
    # 11% last May does not explain a 34% blowout this May
    hist = _history([("2024-05", 0.11), ("2025-05", 0.34)])
    pattern, _ = classify_pattern("2025-05", hist)
    assert pattern is Pattern.ONE_OFF


def test_seasonal_requires_same_direction():
    hist = _history([("2024-12", -0.25), ("2025-12", 0.25)])
    pattern, _ = classify_pattern("2025-12", hist)
    assert pattern is not Pattern.SEASONAL


def test_seasonal_ignores_same_year_months():
    hist = _history([("2025-11", 0.30), ("2025-12", 0.30)])
    pattern, _ = classify_pattern("2025-12", hist)
    assert pattern is not Pattern.SEASONAL


# --- persistence rule ------------------------------------------------------
def test_persistent_fires_when_most_of_window_breaches():
    hist = _history([
        ("2025-01", 0.12), ("2025-02", 0.14), ("2025-03", 0.11),
        ("2025-04", 0.13), ("2025-05", 0.15), ("2025-06", 0.12),
    ])
    pattern, ev = classify_pattern("2025-06", hist)
    assert pattern is Pattern.PERSISTENT
    assert ev["hits"] >= 3


def test_persistent_ignores_opposite_direction_breaches():
    hist = _history([
        ("2025-01", -0.20), ("2025-02", -0.20), ("2025-03", -0.20),
        ("2025-04", 0.01), ("2025-05", 0.01), ("2025-06", 0.12),
    ])
    pattern, _ = classify_pattern("2025-06", hist)
    assert pattern is not Pattern.PERSISTENT


def test_single_breach_is_not_persistent():
    hist = _history([
        ("2025-01", 0.01), ("2025-02", 0.02), ("2025-03", 0.01),
        ("2025-04", 0.00), ("2025-05", 0.01), ("2025-06", 0.30),
    ])
    pattern, _ = classify_pattern("2025-06", hist)
    assert pattern is Pattern.ONE_OFF


# --- accelerating rule -----------------------------------------------------
def test_accelerating_fires_on_widening_gap():
    hist = _history([("2025-04", 0.06), ("2025-05", 0.10), ("2025-06", 0.16)])
    pattern, ev = classify_pattern("2025-06", hist)
    assert pattern is Pattern.ACCELERATING
    assert ev["to_pct"] > ev["from_pct"]


def test_not_accelerating_when_gap_narrows():
    hist = _history([("2025-04", 0.20), ("2025-05", 0.16), ("2025-06", 0.12)])
    pattern, _ = classify_pattern("2025-06", hist)
    assert pattern is not Pattern.ACCELERATING


# --- config ----------------------------------------------------------------
def test_config_changes_classification():
    hist = _history([
        ("2025-01", 0.12), ("2025-02", 0.12), ("2025-03", 0.12),
        ("2025-04", 0.01), ("2025-05", 0.01), ("2025-06", 0.12),
    ])
    strict = CommentaryConfig(persistence_min_count=5)
    assert classify_pattern("2025-06", hist, strict)[0] is not Pattern.PERSISTENT


def test_missing_month_raises():
    with pytest.raises(ValueError, match="not present"):
        classify_pattern("2099-01", _history([("2025-01", 0.1)]))


# --- sentence assembly -----------------------------------------------------
def test_commentary_contains_all_four_parts():
    hist = _history([("2025-05", 0.01), ("2025-06", 0.30)])
    row = pd.Series({
        "month": "2025-06", "department": "Marketing",
        "account": "Advertising & Media", "variance": 45_000,
        "variance_pct": 0.30, "impact": -45_000,
        "ytd_variance": 60_000, "ytd_variance_pct": 0.08,
    })
    text = generate_commentary(row, hist)
    assert "Marketing Advertising & Media" in text   # headline
    assert "30%" in text and "$45K" in text
    assert "one-off" in text                          # pattern
    assert "media and campaign spend" in text         # driver
    assert "YTD" in text                              # context


def test_favourable_variance_is_labelled():
    hist = _history([("2025-06", 0.15)])
    row = pd.Series({
        "month": "2025-06", "department": "Sales", "account": "Revenue",
        "variance": 300_000, "variance_pct": 0.15, "impact": 300_000,
        "ytd_variance": 300_000, "ytd_variance_pct": 0.03,
    })
    assert "favourable" in generate_commentary(row, hist)


# --- end to end ------------------------------------------------------------
def test_add_commentary_on_real_dataset():
    report = build_variance_report(load_data())
    flags = add_commentary(
        flag_variances(report, pct_threshold=0.10, dollar_floor=10_000), report
    )
    assert len(flags) > 0
    assert flags["commentary"].str.len().min() > 50
    assert flags["pattern"].isin([p.value for p in Pattern]).all()


def test_empty_flags_returns_empty_with_columns():
    report = build_variance_report(load_data())
    empty = flag_variances(report, pct_threshold=5.0)
    out = add_commentary(empty, report)
    assert out.empty
    assert "commentary" in out.columns


def test_planted_anomalies_are_all_caught():
    """The six one-off events planted in generate_data.py should all
    surface as flagged items."""
    report = build_variance_report(load_data())
    flags = flag_variances(report, pct_threshold=0.10, dollar_floor=10_000)
    found = set(zip(flags["month"], flags["department"], flags["account"]))
    planted = {
        ("2024-07", "G&A", "Professional Services"),
        ("2024-10", "Marketing", "Events & Sponsorships"),
        ("2025-03", "Operations", "Facilities & Rent"),
        ("2025-05", "Engineering", "Cloud & Infrastructure"),
        ("2025-08", "Customer Success", "Salaries & Benefits"),
    }
    missing = planted - found
    assert not missing, f"planted anomalies not flagged: {missing}"