# Budget vs Actual Variance Dashboard

Automated monthly variance analysis for FP&A: loads budget vs. actual, flags the variances that matter, and writes plain-English commentary explaining each one.

**Live app:** _[link to be added after deployment]_

---

## The problem

Monthly variance analysis is usually a manual, recurring chore. An analyst exports actuals, drops them next to budget in Excel, scans every line for what moved, and then writes a sentence of commentary for each one — often re-explaining the same recurring pattern (a seasonal spike, a persistent departmental overrun) close after close. It's slow, inconsistent between analysts and months, and it's easy to miss a real problem buried in a spreadsheet of hundreds of rows.

## What this does

- Loads a budget-vs-actual dataset (month, department, account, budget, actual).
- Computes variance, % variance, and year-to-date figures.
- Flags the variances that are actually material.
- Generates a plain-English explanation for each flagged item, classified by the kind of story it tells (seasonal, persistent, accelerating, or one-off).
- Presents all of this in an interactive dashboard with filters for department, account, month range, and thresholds.

## Design decisions

This section is the point of the project — the code is straightforward pandas, the judgement calls are what make it useful.

**Favourable vs. unfavourable is not the same as positive vs. negative.**
Revenue coming in over budget is good news; an expense coming in over budget is bad news. The engine computes a signed `impact` column (positive always means good for the P&L) alongside the raw `variance`. If you sorted a "problems" list by raw variance, a large revenue beat would show up as the biggest number on the page — exactly backwards.

**Materiality requires both a percentage threshold and a dollar floor.**
A $400 travel line that's 40% over budget is $160 of noise. A dollar floor alone has the opposite problem — it misses small, fast-growing accounts before they become large ones. An item is only flagged if it breaches both, which mirrors how materiality is judged in real variance reporting.

**Revenue and opex are never summed into one headline number.**
An early version of the dashboard showed "Total Budget: $113M" — $50.9M of revenue plus $62.2M of opex added together. That figure describes nothing real. Worse, in the same version a $788K favourable revenue beat and a $400K unfavourable cost overrun netted against each other into a single "variance," hiding both stories behind a number that looked reassuring. The dashboard now has a P&L view selector (Revenue / Operating expenses / Both, detail only) and never blends the two into one KPI.

**Commentary patterns are classified in a specific order, and that order is the judgement.**
Each flagged item is checked against its own 24-month history and assigned exactly one pattern, first match wins:

1. **Seasonal** — the same calendar month breached in a prior year → a phasing problem, not a new surprise.
2. **Persistent** — most of the trailing months breached in the same direction → the budget assumption itself is probably wrong.
3. **Accelerating** — the gap has been widening for several consecutive months → an early warning worth escalating before it compounds.
4. **One-off** — none of the above → an isolated event.

That ordering is deliberate: a recurring Q4 miss is a conversation about budget phasing, a persistent drift is a conversation about resetting the budget, and an accelerating gap is a conversation about acting now, before month-end. Treating all three the same way ("investigate why we're over budget") would flatten three different actions into one vague sentence.

**The seasonal rule also requires the prior year's breach to be a comparable size, not just present.**
Without that check, a marginal 11% breach in May of last year would "explain away" a genuine 34% blowout in May this year — technically the same month breached twice, but nothing like the same problem. The rule requires the prior year's variance to be at least half the size of the current one before it's allowed to call something seasonal.

**The synthetic dataset is deliberately structured, not random noise.**
It encodes persistent departmental biases (e.g., Engineering contractor spend consistently running over plan while salaries run under, from delayed hiring), real seasonality (Q4-weighted advertising and revenue), and six hardcoded one-off anomalies (a legal dispute, an HVAC repair, a cloud migration overrun, and others). This gives the commentary engine actual patterns to find rather than noise to explain — and lets the test suite assert that every planted anomaly gets caught.

## Running locally

```powershell
# 1. Clone and enter the project
git clone <repo-url>
cd project

# 2. Create and use a virtual environment
python -m venv .venv

# 3. Install dependencies
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

# 4. Generate the synthetic dataset
& ".\.venv\Scripts\python.exe" generate_data.py

# 5. Install dev dependencies and run the test suite
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
& ".\.venv\Scripts\python.exe" -m pytest -q

# 6. Launch the dashboard
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

## Project structure

```
generate_data.py         # Builds the synthetic budget-vs-actual dataset (data/budget_vs_actual.csv)
variance_engine.py       # Variance math: $/% variance, signed impact, YTD, materiality flagging
commentary.py            # Pattern classification and plain-English commentary generation
app.py                   # Streamlit dashboard — loads, filters, and renders; no calculation logic
test_variance_engine.py  # Unit tests for variance_engine.py
test_commentary.py       # Unit tests for commentary.py
requirements.txt         # Direct dependencies
requirements-dev.txt      # Dev-only dependencies (test tooling)
runtime.txt               # Python version pin for Streamlit Cloud
.gitignore
README.md
```

## Testing

41 tests, run with `pytest -q`. They cover:

- Core variance math (dollar/percent variance, zero-budget edge case).
- The favourable/unfavourable sign convention for both revenue and expense accounts.
- YTD accumulation, fiscal-year resets, off-calendar fiscal years, and correctness on unsorted input.
- The materiality rule (percentage alone vs. dollar floor alone vs. both).
- Severity tiers and the unfavourable-only filter.
- Input validation (missing columns, bad account types, duplicate rows, non-numeric values).
- Each commentary pattern rule in isolation (seasonal, persistent, accelerating, one-off), including the seasonal-magnitude check and the persistence direction check.
- Full sentence assembly (headline, pattern, driver, YTD context).
- An end-to-end test against the real generated dataset asserting that **all six planted one-off anomalies are correctly flagged** — the check that the whole pipeline actually catches the problems it was built to catch.

## A note on the data

The dataset is synthetic, generated by `generate_data.py` with a fixed random seed. It is not real company data — it's built to be realistic enough (persistent biases, seasonality, planted anomalies) to give the variance engine and commentary rules something genuine to find.
