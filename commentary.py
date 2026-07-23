"""
commentary.py
-------------
Rule-based, plain-English explanations for flagged budget variances.

DESIGN PHILOSOPHY
Every sentence this module produces can be traced to one rule and one
template. There is no model, no randomness, no hidden state. If a
reviewer asks "why did it say that?", you can point at exactly two
things: the pattern the classifier assigned, and the template that
pattern maps to.

HOW A COMMENT IS BUILT
Each flagged row gets four parts, assembled in order:

    [1 headline]  Marketing Advertising & Media came in 26% ($76K) over
                  budget in Dec 2024.
    [2 pattern ]  This is a recurring Q4 pattern -- the same month
                  breached in 1 prior year.
    [3 driver  ]  Driven primarily by higher media and campaign spend.
    [4 context ]  YTD the account is $181K (7%) over budget.

Parts 2 and 3 are the interesting ones:

  PATTERN CLASSIFICATION (classify_pattern)
    Looks at the item's own 24-month history and assigns exactly one
    label, first match wins. Order matters and is deliberate:

      1. SEASONAL    -- the same calendar month breached in a prior year
                        => it is a phasing problem, not a new surprise
      2. PERSISTENT  -- most of the trailing N months breached the same
                        direction => the budget itself is wrong
      3. ACCELERATING-- breaching and the gap is widening month over
                        month => early warning, worth escalating
      4. ONE_OFF     -- none of the above => isolated event

    That order encodes real FP&A judgement: a recurring seasonal miss is
    a different conversation from a structural overspend, which is
    different again from a single unbudgeted invoice.

  DRIVER PHRASES (ACCOUNT_DRIVERS)
    A lookup from account name to the business language an analyst would
    actually use ("usage-based infrastructure costs" rather than
    "Cloud & Infrastructure was high"). Unmapped accounts fall back to a
    generic phrase, so the module never fails on a new account.

Usage:
    from variance_engine import load_data, build_variance_report, flag_variances
    from commentary import add_commentary

    report = build_variance_report(load_data())
    flags  = flag_variances(report, dollar_floor=10_000)
    flags  = add_commentary(flags, report)
    print(flags[["month", "department", "account", "commentary"]])
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

__all__ = [
    "Pattern",
    "CommentaryConfig",
    "classify_pattern",
    "generate_commentary",
    "add_commentary",
    "ACCOUNT_DRIVERS",
]


class Pattern(str, Enum):
    """The four variance stories. Kept as an Enum so typos fail loudly."""

    SEASONAL = "Seasonal / phasing"
    PERSISTENT = "Persistent trend"
    ACCELERATING = "Accelerating"
    ONE_OFF = "One-off"


@dataclass(frozen=True)
class CommentaryConfig:
    """Thresholds for the classifier. All configurable so the rules stay
    a policy choice rather than a magic number buried in code."""

    pct_threshold: float = 0.10       # must match the flagging threshold
    lookback_months: int = 6          # window for the persistence test
    persistence_share: float = 0.50   # >= this share of lookback breaching
    persistence_min_count: int = 3    # ...and at least this many months
    accel_months: int = 3             # consecutive widening months
    seasonal_similarity: float = 0.50 # prior-year breach must be >= this
                                      # share of the current one's size


# --------------------------------------------------------------------------
# Business language lookup: account -> what actually drives it
# --------------------------------------------------------------------------
ACCOUNT_DRIVERS = {
    "Revenue": ("stronger bookings and deal closure", "softer bookings and deal slippage"),
    "Advertising & Media": ("higher media and campaign spend", "reduced media buying"),
    "Events & Sponsorships": ("additional event and sponsorship commitments", "cancelled or deferred events"),
    "Salaries & Benefits": ("higher headcount or compensation costs", "open roles and slower hiring than planned"),
    "Commissions": ("higher sales attainment", "lower sales attainment"),
    "Contractors": ("increased contractor and agency usage", "reduced contractor usage"),
    "Cloud & Infrastructure": ("higher usage-based infrastructure costs", "lower infrastructure consumption"),
    "Software & Subscriptions": ("additional licences and tooling", "licence consolidation or delayed renewals"),
    "Travel & Entertainment": ("increased travel activity", "reduced travel activity"),
    "Facilities & Rent": ("higher facilities and maintenance costs", "lower facilities costs"),
    "Logistics & Fulfillment": ("higher shipping and fulfilment volumes", "lower fulfilment volumes"),
    "Professional Services": ("higher legal, audit and consulting fees", "lower external advisory spend"),
    "Insurance": ("higher premium renewals", "favourable premium renewals"),
    "Recruiting": ("increased hiring activity and agency fees", "slower hiring activity"),
    "Training & Development": ("additional training programmes", "deferred training spend"),
}
GENERIC_DRIVER = ("higher than planned spend in this account", "lower than planned spend in this account")


def driver_phrase(account: str, over_budget: bool) -> str:
    """Return the business-language driver for an account and direction."""
    over, under = ACCOUNT_DRIVERS.get(account, GENERIC_DRIVER)
    return over if over_budget else under


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def fmt_money(x: float, signed: bool = False) -> str:
    """Format dollars the way a variance report would: $76K, $1.2M.

    Magnitude only by default -- the commentary templates state direction
    in words ("over"/"under"), so a sign there would be redundant. Pass
    signed=True for tables and KPI cards, where the sign is the only thing
    telling the reader which way the variance went.
    """
    a = abs(x)
    prefix = "-" if (signed and x < 0) else ""
    if a >= 1_000_000:
        return f"{prefix}${a / 1_000_000:,.1f}M"
    if a >= 1_000:
        return f"{prefix}${a / 1_000:,.0f}K"
    return f"{prefix}${a:,.0f}"


def fmt_month(month: str) -> str:
    """'2024-12' -> 'Dec 2024'."""
    return pd.Period(month, freq="M").strftime("%b %Y")


def quarter_of(month: str) -> str:
    """'2024-12' -> 'Q4'."""
    return f"Q{(pd.Period(month, freq='M').month - 1) // 3 + 1}"


# --------------------------------------------------------------------------
# Pattern classification
# --------------------------------------------------------------------------
def classify_pattern(
    month: str,
    history: pd.DataFrame,
    config: CommentaryConfig | None = None,
) -> tuple[Pattern, dict]:
    """Classify one flagged month against that item's own history.

    `history` must be the full month-by-month history for a single
    department/account, with columns `month` and `variance_pct`.

    Returns (Pattern, evidence) where `evidence` holds the numbers the
    rule fired on -- so the commentary can quote them and you can audit
    the decision.
    """
    cfg = config or CommentaryConfig()
    h = history.sort_values("month").reset_index(drop=True)

    current = h.loc[h["month"] == month]
    if current.empty:
        raise ValueError(f"{month} not present in the supplied history.")
    cur_pct = float(current["variance_pct"].iloc[0])
    direction = np.sign(cur_pct)

    period = pd.Period(month, freq="M")
    breached = h["variance_pct"].abs() >= cfg.pct_threshold
    same_direction = np.sign(h["variance_pct"]) == direction

    # --- Rule 1: seasonal -- the same calendar month breached in a prior
    # year, AND at comparable magnitude. The magnitude test matters: a
    # marginal 10% breach last May does not explain a 34% blowout this
    # May, and without it a genuine one-off gets excused as "seasonal".
    same_cal_month = h["month"].apply(
        lambda m: pd.Period(m, freq="M").month == period.month
        and pd.Period(m, freq="M").year != period.year
    )
    comparable = h["variance_pct"].abs() >= cfg.seasonal_similarity * abs(cur_pct)
    prior_year_hits = int(
        (same_cal_month & breached & same_direction & comparable).sum()
    )
    if prior_year_hits >= 1:
        return Pattern.SEASONAL, {
            "prior_year_hits": prior_year_hits,
            "quarter": quarter_of(month),
        }

    # --- Rule 2: persistent -- most of the trailing window breached
    window_start = period - cfg.lookback_months
    in_window = h["month"].apply(
        lambda m: window_start <= pd.Period(m, freq="M") <= period
    )
    window = h.loc[in_window]
    hits = int(
        (
            (window["variance_pct"].abs() >= cfg.pct_threshold)
            & (np.sign(window["variance_pct"]) == direction)
        ).sum()
    )
    if (
        hits >= cfg.persistence_min_count
        and len(window) > 0
        and hits / len(window) >= cfg.persistence_share
    ):
        return Pattern.PERSISTENT, {
            "hits": hits,
            "window": len(window),
            "avg_pct": float(window["variance_pct"].mean()),
        }

    # --- Rule 3: accelerating -- gap widening for N consecutive months
    tail = h.loc[h["month"] <= month].tail(cfg.accel_months)
    if len(tail) == cfg.accel_months:
        signed = tail["variance_pct"] * direction
        if (signed.diff().dropna() > 0).all() and (signed > 0).all():
            return Pattern.ACCELERATING, {
                "from_pct": float(tail["variance_pct"].iloc[0]),
                "to_pct": float(tail["variance_pct"].iloc[-1]),
                "months": cfg.accel_months,
            }

    # --- Rule 4: default
    prior = h.loc[h["month"] < month, "variance_pct"]
    return Pattern.ONE_OFF, {
        "typical_pct": float(prior.abs().mean()) if len(prior) else 0.0,
    }


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------
def _pattern_sentence(pattern: Pattern, ev: dict, month: str) -> str:
    """One template per pattern. Read top to bottom in an interview."""
    if pattern is Pattern.SEASONAL:
        yr = "year" if ev["prior_year_hits"] == 1 else "years"
        return (
            f"This is a recurring {ev['quarter']} pattern — the same month "
            f"breached in {ev['prior_year_hits']} prior {yr}, which points to "
            f"budget phasing rather than a new overspend."
        )
    if pattern is Pattern.PERSISTENT:
        return (
            f"This is a persistent trend: {ev['hits']} of the last "
            f"{ev['window']} months breached in the same direction "
            f"(averaging {ev['avg_pct']:+.0%}), suggesting the budget "
            f"assumption itself needs revisiting."
        )
    if pattern is Pattern.ACCELERATING:
        return (
            f"The gap has widened for {ev['months']} consecutive months "
            f"(from {ev['from_pct']:+.0%} to {ev['to_pct']:+.0%}) and warrants "
            f"attention before it compounds further."
        )
    if ev.get("typical_pct", 0) > 0:
        return (
            f"This appears to be a one-off — the account normally runs within "
            f"{ev['typical_pct']:.0%} of budget."
        )
    return "This appears to be an isolated, one-off movement."


def generate_commentary(
    row: pd.Series,
    history: pd.DataFrame,
    config: CommentaryConfig | None = None,
) -> str:
    """Build the full plain-English comment for one flagged row."""
    cfg = config or CommentaryConfig()
    pattern, ev = classify_pattern(row["month"], history, cfg)

    over = row["variance"] > 0
    favourable = row.get("impact", 0) > 0

    # 1. Headline
    headline = (
        f"{row['department']} {row['account']} came in "
        f"{abs(row['variance_pct']):.0%} ({fmt_money(row['variance'])}) "
        f"{'over' if over else 'under'} budget in {fmt_month(row['month'])}"
        f"{' (favourable)' if favourable else ''}."
    )

    # 2. Pattern
    pattern_txt = _pattern_sentence(pattern, ev, row["month"])

    # 3. Driver
    driver_txt = f"Driven primarily by {driver_phrase(row['account'], over)}."

    # 4. YTD context
    parts = [headline, pattern_txt, driver_txt]
    if pd.notna(row.get("ytd_variance")) and pd.notna(row.get("ytd_variance_pct")):
        parts.append(
            f"YTD the account is {fmt_money(row['ytd_variance'])} "
            f"({abs(row['ytd_variance_pct']):.0%}) "
            f"{'over' if row['ytd_variance'] > 0 else 'under'} budget."
        )

    return " ".join(parts)


def add_commentary(
    flags: pd.DataFrame,
    report: pd.DataFrame,
    config: CommentaryConfig | None = None,
) -> pd.DataFrame:
    """Add `pattern` and `commentary` columns to a flagged-items frame.

    `report` is the full variance report (all months), needed because
    classification depends on each item's history, not just the flagged
    month.
    """
    cfg = config or CommentaryConfig()
    if flags.empty:
        return flags.assign(pattern=pd.Series(dtype=str),
                            commentary=pd.Series(dtype=str))

    histories = {
        key: grp[["month", "variance_pct"]]
        for key, grp in report.groupby(["department", "account"], sort=False)
    }

    patterns, comments = [], []
    for _, row in flags.iterrows():
        hist = histories[(row["department"], row["account"])]
        pattern, _ = classify_pattern(row["month"], hist, cfg)
        patterns.append(pattern.value)
        comments.append(generate_commentary(row, hist, cfg))

    out = flags.copy()
    out["pattern"] = patterns
    out["commentary"] = comments
    return out


if __name__ == "__main__":
    from variance_engine import build_variance_report, flag_variances, load_data

    report = build_variance_report(load_data())
    flags = flag_variances(report, pct_threshold=0.10, dollar_floor=10_000)
    flags = add_commentary(flags, report)

    print(f"{len(flags)} flagged items\n")
    print(flags["pattern"].value_counts().to_string(), "\n")
    for _, r in flags.head(6).iterrows():
        print(f"[{r['severity']}] {r['commentary']}\n")
