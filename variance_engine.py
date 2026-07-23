"""
variance_engine.py
------------------
Reusable budget-vs-actual variance calculations for FP&A reporting.

Pure pandas. No Streamlit, no plotting, no file paths hardcoded beyond a
default -- so this module can be imported by the dashboard, a notebook,
a scheduled job, or a test suite.

Core concepts you should be able to explain:

1. VARIANCE vs. IMPACT
   Raw variance is simply actual - budget. But the *business meaning* of
   that number flips depending on the account:
       - Revenue over budget  -> favorable
       - Expense over budget  -> unfavorable
   So every row also gets a signed `impact` (positive = good for the P&L)
   and a `favorability` label. Analysts care about impact; the raw number
   alone will mislead anyone reading the dashboard.

2. MATERIALITY FLOOR
   A $400 travel line coming in 40% over budget is $160 of noise. A $2M
   engineering line 6% over is $120K that matters. Percentage thresholds
   alone over-flag small accounts, so an item must breach BOTH a
   percentage threshold and a dollar floor to be flagged. This is
   standard practice in real variance reporting.

3. YTD RESETS AT THE FISCAL YEAR
   YTD figures accumulate within a fiscal year and reset at year start.
   The default fiscal year is the calendar year; `fiscal_year_start_month`
   supports companies on an off-calendar fiscal year.

Typical use:

    from variance_engine import load_data, build_variance_report, flag_variances

    df = load_data("data/budget_vs_actual.csv")
    report = build_variance_report(df)
    flags = flag_variances(report, pct_threshold=0.10, dollar_floor=25_000)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "VarianceConfig",
    "load_data",
    "compute_variance",
    "add_ytd",
    "build_variance_report",
    "flag_variances",
    "summarize_by",
    "REQUIRED_COLUMNS",
]

REQUIRED_COLUMNS = [
    "month",
    "department",
    "account",
    "account_type",
    "budget",
    "actual",
]

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "budget_vs_actual.csv"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VarianceConfig:
    """Tunable thresholds for flagging. Frozen so it can't be mutated by
    accident halfway through a report run."""

    pct_threshold: float = 0.10        # +/- 10% variance
    dollar_floor: float = 25_000.0     # ignore small-dollar noise
    fiscal_year_start_month: int = 1   # 1 = calendar year


# ---------------------------------------------------------------------------
# Load / validate
# ---------------------------------------------------------------------------
def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the budget-vs-actual CSV and validate its shape.

    Fails loudly on a bad schema rather than producing a silently wrong
    report -- the single most useful habit in finance automation.
    """
    df = pd.read_csv(path)
    return validate(df)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns, coerce dtypes, and reject unusable rows."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required column(s): {missing}")

    out = df.copy()
    out["period"] = pd.PeriodIndex(out["month"], freq="M")
    out["month"] = out["period"].astype(str)
    out["budget"] = pd.to_numeric(out["budget"], errors="coerce")
    out["actual"] = pd.to_numeric(out["actual"], errors="coerce")

    bad = out[["budget", "actual"]].isna().any(axis=1)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} row(s) have non-numeric budget or actual values."
        )

    unknown = set(out["account_type"].unique()) - {"revenue", "expense"}
    if unknown:
        raise ValueError(
            f"account_type must be 'revenue' or 'expense'; found: {sorted(unknown)}"
        )

    dupes = out.duplicated(subset=["month", "department", "account"])
    if dupes.any():
        raise ValueError(
            f"{int(dupes.sum())} duplicate month/department/account row(s) found."
        )

    return out


# ---------------------------------------------------------------------------
# Core variance math
# ---------------------------------------------------------------------------
def compute_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Add monthly $ variance, % variance, signed impact, and favorability.

    variance      = actual - budget           (raw, unsigned meaning)
    variance_pct  = variance / |budget|       (NaN when budget == 0)
    impact        = variance for revenue, -variance for expense
                    -> positive impact is always good for the P&L
    favorability  = 'Favorable' / 'Unfavorable' / 'On budget'
    """
    out = df.copy()
    out["variance"] = out["actual"] - out["budget"]

    denom = out["budget"].abs().replace(0, np.nan)
    out["variance_pct"] = out["variance"] / denom

    sign = np.where(out["account_type"].eq("revenue"), 1.0, -1.0)
    out["impact"] = out["variance"] * sign
    out["impact_pct"] = out["variance_pct"] * sign

    out["favorability"] = np.select(
        [out["impact"] > 0, out["impact"] < 0],
        ["Favorable", "Unfavorable"],
        default="On budget",
    )
    return out


def _fiscal_year(period: pd.PeriodIndex, start_month: int) -> pd.Series:
    """Label each period with its fiscal year.

    For a calendar fiscal year (start_month=1) this is just the year.
    For, say, an April start, Jan-Mar 2025 belongs to FY2024.
    """
    year = period.year.to_numpy()
    month = period.month.to_numpy()
    return pd.Series(np.where(month >= start_month, year, year - 1))


def add_ytd(
    df: pd.DataFrame,
    group_cols: tuple[str, ...] = ("department", "account"),
    fiscal_year_start_month: int = 1,
) -> pd.DataFrame:
    """Add cumulative YTD budget, actual, variance, and YTD % variance.

    Accumulates within each fiscal year for each group (default:
    department + account) and resets at the start of the next year.
    Sorting by period first is what makes cumsum correct -- an
    unsorted frame would silently produce garbage.
    """
    out = df.copy()
    out["fiscal_year"] = _fiscal_year(
        pd.PeriodIndex(out["period"]), fiscal_year_start_month
    ).to_numpy()

    keys = ["fiscal_year", *group_cols]
    out = out.sort_values([*keys, "period"]).reset_index(drop=True)
    g = out.groupby(keys, sort=False)

    out["ytd_budget"] = g["budget"].cumsum()
    out["ytd_actual"] = g["actual"].cumsum()
    out["ytd_variance"] = out["ytd_actual"] - out["ytd_budget"]
    out["ytd_variance_pct"] = out["ytd_variance"] / out["ytd_budget"].abs().replace(0, np.nan)

    sign = np.where(out["account_type"].eq("revenue"), 1.0, -1.0)
    out["ytd_impact"] = out["ytd_variance"] * sign

    return out


def build_variance_report(
    df: pd.DataFrame,
    config: VarianceConfig | None = None,
    group_cols: tuple[str, ...] = ("department", "account"),
) -> pd.DataFrame:
    """Convenience wrapper: variance math + YTD in one call.

    This is the function the dashboard calls on load.
    """
    cfg = config or VarianceConfig()
    out = compute_variance(df)
    out = add_ytd(
        out,
        group_cols=group_cols,
        fiscal_year_start_month=cfg.fiscal_year_start_month,
    )
    return out


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------
def flag_variances(
    df: pd.DataFrame,
    config: VarianceConfig | None = None,
    pct_threshold: float | None = None,
    dollar_floor: float | None = None,
    unfavorable_only: bool = False,
) -> pd.DataFrame:
    """Return rows breaching BOTH the percentage threshold and dollar floor.

    Adds:
        severity      -- 'High' (>= 2x threshold) or 'Medium'
        breach_reason -- 'Over budget' / 'Under budget'

    Explicit pct_threshold / dollar_floor arguments override the config,
    which is what lets the Streamlit sliders drive this without rebuilding
    a config object on every rerun.
    """
    cfg = config or VarianceConfig()
    pct = cfg.pct_threshold if pct_threshold is None else pct_threshold
    floor = cfg.dollar_floor if dollar_floor is None else dollar_floor

    if "variance_pct" not in df.columns:
        raise ValueError("Run compute_variance()/build_variance_report() first.")

    breach = (df["variance_pct"].abs() >= pct) & (df["variance"].abs() >= floor)
    flags = df.loc[breach].copy()

    if unfavorable_only:
        flags = flags[flags["impact"] < 0]

    flags["severity"] = np.where(
        flags["variance_pct"].abs() >= 2 * pct, "High", "Medium"
    )
    flags["breach_reason"] = np.where(
        flags["variance"] > 0, "Over budget", "Under budget"
    )

    return flags.sort_values("impact").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation helper (used by the dashboard's KPI strip and charts)
# ---------------------------------------------------------------------------
def summarize_by(df: pd.DataFrame, *group_cols: str) -> pd.DataFrame:
    """Roll up budget/actual to any grouping and recompute variance.

    Note it recomputes percentages from the summed dollars rather than
    averaging the row-level percentages -- averaging percentages across
    different-sized accounts is a classic reporting error.
    """
    if not group_cols:
        raise ValueError("Provide at least one column to group by.")

    agg = (
        df.groupby(list(group_cols), as_index=False)[["budget", "actual"]]
        .sum()
    )
    agg["variance"] = agg["actual"] - agg["budget"]
    agg["variance_pct"] = agg["variance"] / agg["budget"].abs().replace(0, np.nan)
    return agg


if __name__ == "__main__":
    data = load_data()
    report = build_variance_report(data)
    flagged = flag_variances(report)

    print(f"Rows: {len(report):,}   Flagged: {len(flagged):,} "
          f"({len(flagged) / len(report):.1%})")
    print("\nTop 5 unfavorable variances:")
    cols = ["month", "department", "account", "budget", "actual",
            "variance", "variance_pct", "severity"]
    print(flagged[cols].head(5).to_string(index=False))
