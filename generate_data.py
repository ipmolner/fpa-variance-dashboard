"""
generate_data.py
----------------
Generates a realistic synthetic Budget-vs-Actual dataset for an FP&A
variance dashboard.

Output: data/budget_vs_actual.csv (long format)
Columns: month (YYYY-MM), department, account, account_type, budget, actual

Design (all deliberate, so you can explain it in an interview):

1. STRUCTURE
   - 24 months: Jan 2024 - Dec 2025 (two full calendar years -> clean YTD).
   - 6 departments, each with a realistic set of 4-6 accounts.
   - Revenue lives only under Sales; everything else is expense.

2. BUDGET
   - Each dept/account has an annual "plan" for 2024, grown by a
     dept-specific rate for 2025 (5-12%, i.e. a growing company).
   - The annual plan is spread across months using a seasonality profile
     per account (e.g. advertising is Q4-weighted, recruiting Q1-weighted,
     T&E dips in December). This mimics how a real budget is phased.

3. ACTUALS = budget * (1 + structural bias + seasonal surprise + noise)
   - Structural bias: some dept/accounts run consistently over or under
     (Marketing ads +8%, Engineering contractors +10% while salaries run
     -6% due to hiring delays, etc.). This creates persistent stories.
   - Seasonal surprise: for a few accounts the *actual* seasonality is
     stronger than what was budgeted (Marketing underestimates the Q4
     spike), so the same months flag every year.
   - Noise: N(0, sigma) with sigma of 2.5-4.5% so months wiggle
     realistically instead of being perfectly smooth.

4. ONE-OFF ANOMALIES
   - A short hardcoded list of explainable events (legal fees, a facility
     repair, a cloud migration overrun, a revenue miss from deal slippage).
     Hardcoding keeps them auditable rather than random surprises.

Reproducibility: fixed RNG seed (42).
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

MONTHS = pd.period_range("2024-01", "2025-12", freq="M")

# --------------------------------------------------------------------------
# 1. Department / account structure: annual 2024 plan in dollars
# --------------------------------------------------------------------------
STRUCTURE = {
    "Sales": {
        "Revenue": 24_000_000,
        "Salaries & Benefits": 3_600_000,
        "Commissions": 1_450_000,
        "Travel & Entertainment": 420_000,
        "Software & Subscriptions": 240_000,
    },
    "Marketing": {
        "Salaries & Benefits": 1_800_000,
        "Advertising & Media": 2_400_000,
        "Events & Sponsorships": 600_000,
        "Software & Subscriptions": 300_000,
        "Travel & Entertainment": 180_000,
    },
    "Engineering": {
        "Salaries & Benefits": 6_200_000,
        "Contractors": 900_000,
        "Cloud & Infrastructure": 1_400_000,
        "Software & Subscriptions": 480_000,
        "Travel & Entertainment": 120_000,
    },
    "Operations": {
        "Salaries & Benefits": 2_100_000,
        "Facilities & Rent": 1_100_000,
        "Logistics & Fulfillment": 800_000,
        "Software & Subscriptions": 150_000,
    },
    "Customer Success": {
        "Salaries & Benefits": 1_900_000,
        "Software & Subscriptions": 260_000,
        "Travel & Entertainment": 140_000,
        "Training & Development": 110_000,
    },
    "G&A": {
        "Salaries & Benefits": 1_600_000,
        "Professional Services": 500_000,
        "Insurance": 280_000,
        "Recruiting": 350_000,
        "Facilities & Rent": 400_000,
    },
}

# Annual budget growth for 2025 vs 2024, by department
GROWTH_2025 = {
    "Sales": 0.12,
    "Marketing": 0.10,
    "Engineering": 0.09,
    "Operations": 0.06,
    "Customer Success": 0.08,
    "G&A": 0.05,
}

# --------------------------------------------------------------------------
# 2. Seasonality profiles (12 monthly weights, normalized to sum to 1).
#    These are used to phase the BUDGET across the year.
# --------------------------------------------------------------------------
def _norm(w):
    w = np.asarray(w, dtype=float)
    return w / w.sum()

SEASONALITY = {
    # Q4-heavy with a soft Q1 (typical B2B bookings pattern)
    "Revenue": _norm([0.8, 0.8, 0.9, 0.95, 0.95, 1.0, 0.95, 0.95, 1.05, 1.1, 1.2, 1.35]),
    # Ad spend concentrated in Q4 holiday push, bump in June (mid-year campaign)
    "Advertising & Media": _norm([0.8, 0.8, 0.9, 0.9, 0.95, 1.1, 0.9, 0.9, 1.0, 1.2, 1.35, 1.5]),
    # Big conference seasons: spring and fall
    "Events & Sponsorships": _norm([0.6, 0.7, 1.1, 1.3, 1.2, 0.9, 0.6, 0.7, 1.2, 1.4, 1.3, 1.0]),
    # Commissions follow revenue
    "Commissions": _norm([0.8, 0.8, 0.9, 0.95, 0.95, 1.0, 0.95, 0.95, 1.05, 1.1, 1.2, 1.35]),
    # Travel: dead in Dec/Jan, peaks around conference months
    "Travel & Entertainment": _norm([0.6, 0.9, 1.1, 1.15, 1.15, 1.1, 0.95, 0.9, 1.2, 1.2, 1.05, 0.7]),
    # Recruiting: Q1-weighted (new-year hiring plans, new grad cycle)
    "Recruiting": _norm([1.4, 1.3, 1.2, 1.0, 0.95, 0.9, 0.8, 0.85, 1.0, 0.95, 0.85, 0.8]),
    # Everything else phased flat
}
FLAT = _norm(np.ones(12))

# --------------------------------------------------------------------------
# 3. Structural biases: (department, account) -> persistent actual-vs-budget
#    drift. Positive = consistently over budget. These create the stories.
# --------------------------------------------------------------------------
BIAS = {
    ("Marketing", "Advertising & Media"): 0.08,     # always spends past plan
    ("Marketing", "Events & Sponsorships"): 0.05,
    ("Engineering", "Contractors"): 0.10,           # backfilling unfilled roles...
    ("Engineering", "Salaries & Benefits"): -0.06,  # ...because hiring is behind plan
    ("Engineering", "Cloud & Infrastructure"): 0.04, # usage-based costs creep
    ("Sales", "Revenue"): 0.02,                     # slight overperformance
    ("Sales", "Commissions"): 0.04,                 # follows revenue beat
    ("Customer Success", "Salaries & Benefits"): -0.03,
    ("Operations", "Logistics & Fulfillment"): 0.05,
    ("G&A", "Professional Services"): 0.06,         # legal/audit always runs hot
}
DEFAULT_BIAS = 0.0

# Accounts where ACTUAL seasonality exceeds what was budgeted:
# extra multiplier applied to actuals in specific months (0-indexed Jan=0).
SEASONAL_SURPRISE = {
    ("Marketing", "Advertising & Media"): {9: 0.04, 10: 0.07, 11: 0.10},  # Q4 blowout
    ("Sales", "Travel & Entertainment"): {8: 0.06, 9: 0.06},              # fall conf. season
}

# Noise level (std dev of monthly wiggle) by account type
def noise_sigma(account: str) -> float:
    if account in ("Salaries & Benefits", "Facilities & Rent", "Insurance"):
        return 0.025   # headcount/contract driven -> stable
    if account == "Revenue":
        return 0.035
    return 0.045       # discretionary spend wiggles more

# --------------------------------------------------------------------------
# 4. One-off anomalies: (month, dept, account) -> (pct shock, reason)
#    The reason column is NOT exported; it's documentation for you.
# --------------------------------------------------------------------------
ANOMALIES = {
    ("2024-04", "Sales", "Revenue"): (-0.12, "Two enterprise deals slipped to Q3"),
    ("2024-07", "G&A", "Professional Services"): (0.55, "Unbudgeted legal fees (contract dispute)"),
    ("2024-10", "Marketing", "Events & Sponsorships"): (0.40, "Late decision to sponsor industry conference"),
    ("2025-03", "Operations", "Facilities & Rent"): (0.35, "Emergency HVAC repair at HQ"),
    ("2025-05", "Engineering", "Cloud & Infrastructure"): (0.30, "Data-platform migration overrun"),
    ("2025-08", "Customer Success", "Salaries & Benefits"): (0.15, "Severance from team restructuring"),
}

# --------------------------------------------------------------------------
# Build the dataset
# --------------------------------------------------------------------------
def build() -> pd.DataFrame:
    rows = []
    for dept, accounts in STRUCTURE.items():
        for account, annual_2024 in accounts.items():
            weights = SEASONALITY.get(account, FLAT)
            annual = {2024: annual_2024,
                      2025: annual_2024 * (1 + GROWTH_2025[dept])}
            bias = BIAS.get((dept, account), DEFAULT_BIAS)
            surprises = SEASONAL_SURPRISE.get((dept, account), {})
            sigma = noise_sigma(account)

            for m in MONTHS:
                budget = annual[m.year] * weights[m.month - 1]
                shock = ANOMALIES.get((str(m), dept, account), (0.0, None))[0]
                noise = rng.normal(0, sigma)
                surprise = surprises.get(m.month - 1, 0.0)
                actual = budget * (1 + bias + surprise + noise + shock)

                rows.append({
                    "month": str(m),
                    "department": dept,
                    "account": account,
                    "account_type": "revenue" if account == "Revenue" else "expense",
                    "budget": round(budget, 0),
                    "actual": round(actual, 0),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    out = Path(__file__).parent / "data" / "budget_vs_actual.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out}")
    print(f"{df['department'].nunique()} departments, "
          f"{df.groupby('department')['account'].nunique().sum()} dept-account pairs, "
          f"{df['month'].nunique()} months")
