"""
test_variance_engine.py
-----------------------
Unit tests for variance_engine.py. Run with:  pytest -q

These are deliberately small and readable -- each one asserts a single
business rule you'd be asked about in an interview.
"""

import numpy as np
import pandas as pd
import pytest

from variance_engine import (
    VarianceConfig,
    add_ytd,
    build_variance_report,
    compute_variance,
    flag_variances,
    summarize_by,
    validate,
)


def _frame(rows):
    return validate(pd.DataFrame(rows))


def _row(month, dept="Marketing", account="Advertising & Media",
         atype="expense", budget=100_000, actual=100_000):
    return {"month": month, "department": dept, "account": account,
            "account_type": atype, "budget": budget, "actual": actual}


# --- basic math ------------------------------------------------------------
def test_variance_dollars_and_pct():
    df = _frame([_row("2024-01", budget=100_000, actual=112_000)])
    out = compute_variance(df)
    assert out.loc[0, "variance"] == 12_000
    assert out.loc[0, "variance_pct"] == pytest.approx(0.12)


def test_zero_budget_gives_nan_not_inf():
    df = _frame([_row("2024-01", budget=0, actual=5_000)])
    out = compute_variance(df)
    assert np.isnan(out.loc[0, "variance_pct"])
    assert out.loc[0, "variance"] == 5_000


# --- favorability sign convention -----------------------------------------
def test_expense_over_budget_is_unfavorable():
    df = _frame([_row("2024-01", atype="expense", budget=100, actual=110)])
    out = compute_variance(df)
    assert out.loc[0, "impact"] == -10
    assert out.loc[0, "favorability"] == "Unfavorable"


def test_revenue_over_budget_is_favorable():
    df = _frame([_row("2024-01", dept="Sales", account="Revenue",
                      atype="revenue", budget=100, actual=110)])
    out = compute_variance(df)
    assert out.loc[0, "impact"] == 10
    assert out.loc[0, "favorability"] == "Favorable"


def test_revenue_under_budget_is_unfavorable():
    df = _frame([_row("2024-01", dept="Sales", account="Revenue",
                      atype="revenue", budget=100, actual=90)])
    out = compute_variance(df)
    assert out.loc[0, "favorability"] == "Unfavorable"


# --- YTD -------------------------------------------------------------------
def test_ytd_accumulates_within_year():
    df = _frame([
        _row("2024-01", budget=100, actual=110),
        _row("2024-02", budget=100, actual=120),
        _row("2024-03", budget=100, actual=90),
    ])
    out = add_ytd(compute_variance(df)).sort_values("month")
    assert list(out["ytd_actual"]) == [110, 230, 320]
    assert list(out["ytd_budget"]) == [100, 200, 300]
    assert out.iloc[-1]["ytd_variance"] == 20


def test_ytd_resets_at_new_fiscal_year():
    df = _frame([
        _row("2024-12", budget=100, actual=150),
        _row("2025-01", budget=100, actual=105),
    ])
    out = add_ytd(compute_variance(df)).sort_values("month")
    assert out.iloc[-1]["ytd_actual"] == 105  # not 255


def test_ytd_respects_offset_fiscal_year():
    # FY starts in April: Mar-2025 still belongs to FY2024
    df = _frame([
        _row("2025-03", budget=100, actual=100),
        _row("2025-04", budget=100, actual=100),
    ])
    out = add_ytd(compute_variance(df), fiscal_year_start_month=4)
    fy = dict(zip(out["month"], out["fiscal_year"]))
    assert fy["2025-03"] == 2024
    assert fy["2025-04"] == 2025


def test_ytd_is_correct_even_if_input_unsorted():
    df = _frame([
        _row("2024-03", budget=100, actual=90),
        _row("2024-01", budget=100, actual=110),
        _row("2024-02", budget=100, actual=120),
    ])
    out = add_ytd(compute_variance(df)).sort_values("month")
    assert list(out["ytd_actual"]) == [110, 230, 320]


def test_ytd_separates_groups():
    df = _frame([
        _row("2024-01", dept="Marketing", budget=100, actual=110),
        _row("2024-01", dept="Engineering", account="Contractors",
             budget=100, actual=200),
    ])
    out = add_ytd(compute_variance(df))
    assert set(out["ytd_actual"]) == {110, 200}


# --- flagging --------------------------------------------------------------
def test_flag_requires_both_pct_and_dollar_breach():
    df = _frame([
        # 40% over but only $160 -> immaterial, should NOT flag
        _row("2024-01", account="Travel & Entertainment", budget=400, actual=560),
        # 12% over and $120k -> should flag
        _row("2024-02", account="Cloud & Infrastructure",
             budget=1_000_000, actual=1_120_000),
    ])
    flags = flag_variances(compute_variance(df),
                           pct_threshold=0.10, dollar_floor=25_000)
    assert len(flags) == 1
    assert flags.loc[0, "account"] == "Cloud & Infrastructure"


def test_severity_high_at_double_threshold():
    df = _frame([
        _row("2024-01", budget=100_000, actual=112_000),  # 12% -> Medium
        _row("2024-02", budget=100_000, actual=135_000),  # 35% -> High
    ])
    flags = flag_variances(compute_variance(df),
                           pct_threshold=0.10, dollar_floor=1_000)
    sev = dict(zip(flags["month"], flags["severity"]))
    assert sev["2024-01"] == "Medium"
    assert sev["2024-02"] == "High"


def test_unfavorable_only_filter():
    df = _frame([
        _row("2024-01", budget=100_000, actual=130_000),  # expense over -> bad
        _row("2024-02", budget=100_000, actual=70_000),   # expense under -> good
    ])
    flags = flag_variances(compute_variance(df), pct_threshold=0.10,
                           dollar_floor=1_000, unfavorable_only=True)
    assert len(flags) == 1
    assert flags.loc[0, "month"] == "2024-01"


def test_threshold_argument_overrides_config():
    df = _frame([_row("2024-01", budget=100_000, actual=106_000)])
    report = compute_variance(df)
    cfg = VarianceConfig(pct_threshold=0.10, dollar_floor=1_000)
    assert len(flag_variances(report, config=cfg)) == 0
    assert len(flag_variances(report, config=cfg, pct_threshold=0.05)) == 1


# --- aggregation -----------------------------------------------------------
def test_summarize_recomputes_pct_from_totals():
    # Averaging the row percentages would give (100% + 1%)/2 = 50.5%.
    # The correct answer is total variance / total budget:
    #   ($1,000 + $10,000) / $1,001,000 = ~1.1%
    df = _frame([
        _row("2024-01", account="Small", budget=1_000, actual=2_000),
        _row("2024-01", account="Large", budget=1_000_000, actual=1_010_000),
    ])
    agg = summarize_by(compute_variance(df), "department")
    assert agg.loc[0, "variance_pct"] == pytest.approx(11_000 / 1_001_000)


# --- validation ------------------------------------------------------------
def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing required column"):
        validate(pd.DataFrame({"month": ["2024-01"], "budget": [1]}))


def test_bad_account_type_raises():
    with pytest.raises(ValueError, match="account_type"):
        _frame([_row("2024-01", atype="asset")])


def test_duplicate_rows_raise():
    with pytest.raises(ValueError, match="duplicate"):
        _frame([_row("2024-01"), _row("2024-01")])


def test_non_numeric_amounts_raise():
    with pytest.raises(ValueError, match="non-numeric"):
        _frame([_row("2024-01", budget="n/a")])


def test_bare_year_is_rejected():
    # pandas would silently turn "2025" into 2025-01; that is a wrong
    # answer, not a convenience.
    with pytest.raises(ValueError, match="bare year"):
        _frame([_row("2025")])


def test_unparseable_month_raises_clearly():
    with pytest.raises(ValueError, match="could not be parsed"):
        _frame([_row("not-a-month")])


def test_common_month_formats_are_accepted():
    # Being lenient here is genuinely useful for uploaded files.
    for fmt in ("2024-01", "Jan2024", "01/2024"):
        assert _frame([_row(fmt)]).loc[0, "month"] == "2024-01"


# --- end-to-end on the real file ------------------------------------------
def test_report_runs_on_generated_dataset():
    from variance_engine import load_data
    report = build_variance_report(load_data())
    assert len(report) == 672
    assert report["ytd_budget"].notna().all()
    # every fiscal year should have 12 months per dept/account
    counts = report.groupby(["fiscal_year", "department", "account"]).size()
    assert set(counts.unique()) == {12}
