"""
app.py
------
Streamlit dashboard for the Budget-vs-Actual variance report.

All calculation logic lives in variance_engine.py / commentary.py, and all
Excel formatting in exports.py. This module only loads, filters, and
renders — nothing here recomputes a number those modules already produce.

Two design notes worth being able to explain:

1. Revenue and expense are never summed into a single headline figure.
   Adding a $50.9M revenue plan to a $62.2M opex plan produces "$113M of
   budget", which represents nothing, and netting a favourable revenue
   beat against an unfavourable cost overrun hides both. The P&L view
   selector keeps them separate.

2. YTD figures always accumulate from the start of the fiscal year, not
   from the start of the selected month range. Year-to-date means
   year-to-date; re-basing it to an arbitrary filter would produce a
   number no one could reconcile to the ledger.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from commentary import add_commentary, fmt_money
from exports import build_excel_report, template_csv_bytes
from variance_engine import (
    DEFAULT_DATA_PATH,
    build_variance_report,
    flag_variances,
    load_data,
    summarize_by,
    validate,
)

st.set_page_config(page_title="Budget vs Actual Variance Dashboard", layout="wide")

FAVOURABLE = "#2ca02c"
UNFAVOURABLE = "#d62728"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_bundled():
    return build_variance_report(load_data(DEFAULT_DATA_PATH))


@st.cache_data(show_spinner=False)
def load_uploaded(file_bytes: bytes):
    """Validate and process an uploaded CSV.

    Cached on the file's bytes, so re-running filters doesn't re-parse.
    Validation errors surface as ValueError and are caught by the caller —
    a bad upload should produce a clear message, never a silent wrong
    number.
    """
    from io import BytesIO

    raw = pd.read_csv(BytesIO(file_bytes))
    return build_variance_report(validate(raw))


st.title("Budget vs Actual Variance Dashboard")

# ---------------------------------------------------------------------------
# Sidebar: data source
# ---------------------------------------------------------------------------
st.sidebar.header("Data source")

uploaded = st.sidebar.file_uploader(
    "Upload budget vs actual CSV",
    type="csv",
    help="Leave empty to use the bundled synthetic dataset.",
)
st.sidebar.download_button(
    "Download CSV template",
    data=template_csv_bytes(),
    file_name="budget_vs_actual_template.csv",
    mime="text/csv",
    help="Required columns: month, department, account, account_type, budget, actual",
)

report = None
if uploaded is not None:
    try:
        report = load_uploaded(uploaded.getvalue())
        st.sidebar.success(f"Loaded {len(report):,} rows from {uploaded.name}")
        source_label = uploaded.name
    except Exception as exc:  # validation is deliberately loud
        st.sidebar.error(f"Could not read that file:\n\n{exc}")
        st.error(
            "The uploaded file could not be processed. Download the template "
            "above to see the expected schema, then try again."
        )
        st.stop()
else:
    if not Path(DEFAULT_DATA_PATH).exists():
        st.error(
            f"Data file not found at `{DEFAULT_DATA_PATH}`. Run "
            "`generate_data.py` to create it, or upload a CSV in the sidebar."
        )
        st.stop()
    report = load_bundled()
    source_label = "bundled synthetic dataset"

st.caption(
    f"Source: {source_label} · {len(report):,} rows · "
    f"{report['month'].min()} to {report['month'].max()}"
)

# ---------------------------------------------------------------------------
# Sidebar: filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

available_types = sorted(report["account_type"].unique())
view_options = []
if "expense" in available_types:
    view_options.append("Operating expenses")
if "revenue" in available_types:
    view_options.append("Revenue")
if len(available_types) > 1:
    view_options.append("Both (detail only)")

view = st.sidebar.radio(
    "P&L view",
    view_options,
    index=0,
    help=(
        "Revenue and expense are reported separately. Summing them into one "
        "figure would be meaningless, so KPIs and charts follow this choice."
    ),
)
TYPE_MAP = {"Operating expenses": ["expense"], "Revenue": ["revenue"]}
selected_types = TYPE_MAP.get(view, available_types)
single_view = view != "Both (detail only)"

departments = sorted(report["department"].unique())
accounts = sorted(
    report.loc[report["account_type"].isin(selected_types), "account"].unique()
)
months = sorted(report["month"].unique())

selected_departments = st.sidebar.multiselect(
    "Department", departments, default=departments
)
selected_accounts = st.sidebar.multiselect("Account", accounts, default=accounts)

if len(months) > 1:
    month_start, month_end = st.sidebar.select_slider(
        "Month range", options=months, value=(months[0], months[-1])
    )
else:
    month_start = month_end = months[0]

pct_threshold = st.sidebar.slider(
    "Variance threshold (%)", min_value=5, max_value=25, value=10, step=1
) / 100.0
dollar_floor = st.sidebar.number_input(
    "Dollar floor ($)", min_value=0, value=10_000, step=5_000
)
st.sidebar.caption(
    "An item must breach **both** the variance % threshold and the dollar "
    "floor to be flagged. Percentage alone over-flags small accounts; a "
    "dollar floor alone misses fast-growing small ones."
)

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
mask = (
    report["account_type"].isin(selected_types)
    & report["department"].isin(selected_departments)
    & report["account"].isin(selected_accounts)
    & (report["month"] >= month_start)
    & (report["month"] <= month_end)
)
filtered = report.loc[mask].copy()

if filtered.empty:
    st.warning("No data matches the current filters. Widen the selection to continue.")
    st.stop()

flagged = add_commentary(
    flag_variances(filtered, pct_threshold=pct_threshold, dollar_floor=dollar_floor),
    report,
)

# ---------------------------------------------------------------------------
# 1. KPI row
# ---------------------------------------------------------------------------
if not single_view:
    st.info(
        "Showing revenue and expense together. Headline totals are suppressed "
        "because summing revenue and cost produces a figure with no meaning — "
        "select a single view for KPIs and charts."
    )
    k1, k2 = st.columns(2)
    k1.metric("Line items in view", f"{len(filtered):,}")
    k2.metric("Flagged items", f"{len(flagged):,}")
else:
    total_budget = filtered["budget"].sum()
    total_actual = filtered["actual"].sum()
    net_variance = total_actual - total_budget
    net_pct = net_variance / abs(total_budget) if total_budget else 0.0
    net_impact = filtered["impact"].sum()

    label = "Revenue" if view == "Revenue" else "Opex"
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{label} Budget", fmt_money(total_budget))
    k2.metric(f"{label} Actual", fmt_money(total_actual))
    k3.metric(
        "Variance vs Budget",
        fmt_money(net_variance, signed=True),
        delta=f"{net_pct:+.1%} ({'favourable' if net_impact >= 0 else 'unfavourable'})",
        delta_color="normal" if net_impact >= 0 else "inverse",
    )
    k4.metric("Flagged Items", f"{len(flagged):,}")

st.divider()

# ---------------------------------------------------------------------------
# 2. Tabs: Monthly / Year-to-date / Bridge
# ---------------------------------------------------------------------------
tab_month, tab_ytd, tab_bridge = st.tabs(
    ["Monthly trend", "Year-to-date", "Budget-to-actual bridge"]
)

with tab_month:
    if not single_view:
        st.caption("Select a single P&L view to see the trend.")
    else:
        trend = summarize_by(filtered, "month").sort_values("month")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend["month"], y=trend["budget"], mode="lines+markers",
            name="Budget", line=dict(dash="dash", color="#7f7f7f")))
        fig.add_trace(go.Scatter(
            x=trend["month"], y=trend["actual"], mode="lines+markers",
            name="Actual", line=dict(color="#1f3864")))
        fig.update_layout(
            xaxis_title="Month", yaxis_title="Dollars", hovermode="x unified",
            margin=dict(t=20, b=20), height=400,
        )
        st.plotly_chart(fig, width="stretch")

with tab_ytd:
    if not single_view:
        st.caption("Select a single P&L view to see the year-to-date position.")
    else:
        st.caption(
            "Cumulative within each fiscal year, resetting in January. YTD "
            "always accumulates from the start of the year, not from the "
            "start of the selected month range."
        )
        ytd = (
            filtered.groupby(["fiscal_year", "month"], as_index=False)
            [["ytd_budget", "ytd_actual", "ytd_variance", "ytd_impact"]].sum()
            .sort_values("month")
        )

        latest = ytd.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"YTD Budget (as of {latest['month']})",
                  fmt_money(latest["ytd_budget"]))
        c2.metric("YTD Actual", fmt_money(latest["ytd_actual"]))
        ytd_pct = (latest["ytd_variance"] / latest["ytd_budget"]
                   if latest["ytd_budget"] else 0.0)
        c3.metric(
            "YTD Variance",
            fmt_money(latest["ytd_variance"], signed=True),
            delta=f"{ytd_pct:+.1%}",
            delta_color="normal" if latest["ytd_impact"] >= 0 else "inverse",
        )

        fig = go.Figure()
        for fy, grp in ytd.groupby("fiscal_year"):
            fig.add_trace(go.Scatter(
                x=grp["month"], y=grp["ytd_budget"], mode="lines",
                name=f"FY{fy} Budget", line=dict(dash="dash", color="#7f7f7f"),
                legendgroup=str(fy)))
            fig.add_trace(go.Scatter(
                x=grp["month"], y=grp["ytd_actual"], mode="lines",
                name=f"FY{fy} Actual", line=dict(color="#1f3864"),
                legendgroup=str(fy), fill="tonexty",
                fillcolor="rgba(214,39,40,0.12)"))
        fig.update_layout(
            xaxis_title="Month", yaxis_title="Cumulative dollars",
            hovermode="x unified", margin=dict(t=20, b=20), height=400,
        )
        st.plotly_chart(fig, width="stretch")

with tab_bridge:
    if not single_view:
        st.caption("Select a single P&L view to see the bridge.")
    else:
        st.caption(
            "How total budget became total actual, with each department's "
            "variance as a step. Steps are raw variance (actual − budget), so "
            "a bar rising means more dollars, which is unfavourable for opex "
            "and favourable for revenue."
        )
        dept = (
            filtered.groupby("department", as_index=False)[["budget", "actual"]]
            .sum()
        )
        dept["variance"] = dept["actual"] - dept["budget"]
        dept = dept.sort_values("variance", ascending=False)

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * len(dept) + ["total"],
            x=["Budget"] + dept["department"].tolist() + ["Actual"],
            y=[filtered["budget"].sum()] + dept["variance"].tolist() + [0],
            text=[fmt_money(filtered["budget"].sum())]
                 + [fmt_money(v, signed=True) for v in dept["variance"]]
                 + [fmt_money(filtered["actual"].sum())],
            textposition="outside",
            connector=dict(line=dict(color="#bbbbbb")),
            increasing=dict(marker=dict(color="#d62728")),
            decreasing=dict(marker=dict(color="#2ca02c")),
            totals=dict(marker=dict(color="#1f3864")),
        ))
        fig.update_layout(
            yaxis_title="Dollars", margin=dict(t=40, b=20), height=440,
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 3. P&L impact by department
# ---------------------------------------------------------------------------
st.subheader("P&L Impact by Department")
st.caption(
    "Positive = favourable. Revenue above plan and spend below plan both "
    "count as favourable, which is why this uses signed impact rather than "
    "raw variance."
)

by_dept = (
    filtered.groupby("department", as_index=False)["impact"].sum()
    .sort_values("impact")
)
fig_dept = go.Figure(go.Bar(
    x=by_dept["impact"], y=by_dept["department"], orientation="h",
    marker_color=[FAVOURABLE if v >= 0 else UNFAVOURABLE for v in by_dept["impact"]],
    text=[fmt_money(v, signed=True) for v in by_dept["impact"]],
    textposition="auto",
))
fig_dept.update_layout(
    xaxis_title="P&L impact ($) — positive is favourable", yaxis_title="",
    margin=dict(t=20, b=20), height=90 + 45 * len(by_dept),
)
st.plotly_chart(fig_dept, width="stretch")

# ---------------------------------------------------------------------------
# 4. Flagged items
# ---------------------------------------------------------------------------
st.subheader("Flagged Items")

if flagged.empty:
    st.info(
        "No items breach both thresholds under the current filters. Lower the "
        "variance threshold or the dollar floor to widen the net."
    )
else:
    st.caption(" · ".join(
        f"**{k}**: {v}" for k, v in flagged["pattern"].value_counts().items()
    ))

    display_cols = [
        "month", "department", "account", "budget", "actual", "variance",
        "variance_pct", "severity", "pattern", "commentary",
    ]
    table = flagged.sort_values("impact")[display_cols].copy()

    shown = table.copy()
    for col in ("budget", "actual"):
        shown[col] = shown[col].apply(fmt_money)
    shown["variance"] = shown["variance"].apply(lambda v: fmt_money(v, signed=True))
    shown["variance_pct"] = shown["variance_pct"].apply(lambda v: f"{v:+.1%}")

    st.dataframe(
        shown, width="stretch", hide_index=True,
        column_config={
            "month": st.column_config.TextColumn("Month", width="small"),
            "variance_pct": st.column_config.TextColumn("Var %", width="small"),
            "commentary": st.column_config.TextColumn("Commentary", width="large"),
        },
    )

# ---------------------------------------------------------------------------
# 5. Exports
# ---------------------------------------------------------------------------
st.subheader("Export")
e1, e2 = st.columns(2)

with e1:
    st.download_button(
        "Download Excel report (.xlsx)",
        data=build_excel_report(
            filtered, flagged, view, pct_threshold, float(dollar_floor)
        ),
        file_name="variance_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Summary, flagged items with commentary, and full detail — formatted.",
        width="stretch",
    )

with e2:
    csv_source = flagged if not flagged.empty else filtered
    st.download_button(
        "Download flagged items (.csv)",
        data=csv_source.to_csv(index=False).encode("utf-8"),
        file_name="flagged_variances.csv",
        mime="text/csv",
        help="Raw unformatted values for further analysis.",
        width="stretch",
    )
