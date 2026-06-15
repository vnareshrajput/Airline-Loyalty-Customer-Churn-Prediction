"""
charts.py — Plotly chart factory
All charts share the enterprise dark theme.
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
PALETTE = {
    "Aurora": "#63b3ed",
    "Nova":   "#9f7aea",
    "Star":   "#68d391",
}
SEQUENTIAL  = "Blues"
DIVERGING   = "RdBu"
QUAL_COLORS = ["#63b3ed", "#9f7aea", "#68d391", "#f6ad55", "#fc8181", "#4fd1c5", "#f687b3"]

BASE_LAYOUT = dict(
    paper_bgcolor = "rgba(10,14,26,0)",
    plot_bgcolor  = "rgba(10,14,26,0)",
    font          = dict(family="Inter, sans-serif", color="#a0aec0", size=12),
    margin        = dict(l=40, r=20, t=50, b=40),
    legend        = dict(bgcolor="rgba(13,21,38,0.8)", bordercolor="rgba(99,179,237,0.2)",
                         borderwidth=1, font=dict(size=11)),
    xaxis         = dict(gridcolor="rgba(99,179,237,0.08)", zerolinecolor="rgba(99,179,237,0.15)"),
    yaxis         = dict(gridcolor="rgba(99,179,237,0.08)", zerolinecolor="rgba(99,179,237,0.15)"),
)

def _apply_theme(fig, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text=title, font=dict(size=15, color="#e2e8f0"), x=0.01),
        height=height,
    )
    return fig


# --------------------------------------------------------------------------- #
# Line / Area
# --------------------------------------------------------------------------- #
def revenue_trend_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Revenue"],
        mode="lines", name="Revenue",
        line=dict(color="#63b3ed", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(99,179,237,0.08)",
    ))
    return _apply_theme(fig, "Monthly Revenue Trend", 350)


# --------------------------------------------------------------------------- #
# Bar
# --------------------------------------------------------------------------- #
def tier_revenue_bar(tier_df: pd.DataFrame) -> go.Figure:
    colors = [PALETTE.get(t, "#63b3ed") for t in tier_df["Tier"]]
    fig = go.Figure(go.Bar(
        x=tier_df["Tier"], y=tier_df["Revenue"],
        marker_color=colors,
        text=[f"${v/1e6:.1f}M" for v in tier_df["Revenue"]],
        textposition="outside",
    ))
    return _apply_theme(fig, "Revenue by Loyalty Tier", 350)


def province_revenue_bar(prov_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=prov_df["Revenue"], y=prov_df["Province"],
        orientation="h",
        marker_color=QUAL_COLORS[0],
        text=[f"${v/1e6:.1f}M" for v in prov_df["Revenue"]],
        textposition="outside",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _apply_theme(fig, "Revenue by Province", 400)


# --------------------------------------------------------------------------- #
# Pie / Donut
# --------------------------------------------------------------------------- #
def tier_customer_donut(tier_df: pd.DataFrame) -> go.Figure:
    colors = [PALETTE.get(t, "#63b3ed") for t in tier_df["Tier"]]
    fig = go.Figure(go.Pie(
        labels=tier_df["Tier"], values=tier_df["Customers"],
        hole=0.55,
        marker_colors=colors,
        textinfo="label+percent",
        textfont_size=12,
    ))
    return _apply_theme(fig, "Customer Distribution by Tier", 350)


def gender_donut(cdf: pd.DataFrame) -> go.Figure:
    counts = cdf["Gender"].value_counts().reset_index()
    counts.columns = ["Gender", "Count"]
    fig = go.Figure(go.Pie(
        labels=counts["Gender"], values=counts["Count"],
        hole=0.5,
        marker_colors=["#9f7aea", "#63b3ed"],
        textinfo="label+percent",
    ))
    return _apply_theme(fig, "Gender Split", 320)


# --------------------------------------------------------------------------- #
# Histogram / Box / Violin
# --------------------------------------------------------------------------- #
def clv_histogram(cdf: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        cdf, x="CLV", nbins=60, color="Loyalty_Card",
        color_discrete_map=PALETTE,
        barmode="overlay", opacity=0.7,
        labels={"CLV": "Customer Lifetime Value ($)", "Loyalty_Card": "Tier"},
    )
    return _apply_theme(fig, "CLV Distribution by Tier", 380)


def salary_violin(cdf: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for tier, color in PALETTE.items():
        sub = cdf[cdf["Loyalty_Card"] == tier]
        fig.add_trace(go.Violin(
            y=sub["Salary"], name=tier, box_visible=True,
            meanline_visible=True, fillcolor=color.replace(")", ",0.2)").replace("rgb","rgba"),
            line_color=color, opacity=0.8,
        ))
    return _apply_theme(fig, "Salary Distribution by Tier", 400)


def engagement_box(cdf: pd.DataFrame) -> go.Figure:
    fig = px.box(
        cdf, x="Loyalty_Card", y="Engagement",
        color="Loyalty_Card", color_discrete_map=PALETTE,
        labels={"Loyalty_Card": "Tier", "Engagement": "Engagement Score"},
        points="outliers",
    )
    return _apply_theme(fig, "Engagement Score Distribution", 380)


# --------------------------------------------------------------------------- #
# Heatmap
# --------------------------------------------------------------------------- #
def correlation_heatmap(cdf: pd.DataFrame) -> go.Figure:
    numeric_cols = ["CLV", "Total_Flights", "Total_Distance", "Total_Revenue",
                    "Salary", "Engagement", "Churn_Risk_Score", "Months_Active"]
    corr = cdf[numeric_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0, text=np.round(corr.values, 2),
        texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="r"),
    ))
    return _apply_theme(fig, "Correlation Matrix", 480)


def province_flight_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = (
        df.groupby(["Province", "Loyalty Card"])["Total Flights"]
          .sum()
          .unstack(fill_value=0)
    )
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="Blues",
        text=pivot.values.astype(int),
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    return _apply_theme(fig, "Total Flights: Province × Tier", 520)


# --------------------------------------------------------------------------- #
# Scatter
# --------------------------------------------------------------------------- #
def clv_vs_flights_scatter(cdf: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        cdf.sample(min(3000, len(cdf)), random_state=42),
        x="Total_Flights", y="CLV",
        color="Loyalty_Card", color_discrete_map=PALETTE,
        opacity=0.55, size_max=8,
        labels={"Total_Flights": "Total Flights", "Loyalty_Card": "Tier"},
        trendline="lowess",
    )
    return _apply_theme(fig, "CLV vs Total Flights", 420)


def engagement_vs_redemption(cdf: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        cdf.sample(min(3000, len(cdf)), random_state=1),
        x="Redemption_Rate", y="Engagement",
        color="Loyalty_Card", color_discrete_map=PALETTE,
        opacity=0.5,
        labels={"Redemption_Rate": "Redemption Rate", "Loyalty_Card": "Tier"},
    )
    return _apply_theme(fig, "Engagement vs Redemption Rate", 400)


# --------------------------------------------------------------------------- #
# Treemap
# --------------------------------------------------------------------------- #
def province_treemap(cdf: pd.DataFrame) -> go.Figure:
    grp = (
        cdf.groupby(["Province", "Loyalty_Card"])
           .agg(Revenue=("Total_Revenue", "sum"), Customers=("Loyalty Number", "count"))
           .reset_index()
    )
    fig = px.treemap(
        grp, path=[px.Constant("Canada"), "Province", "Loyalty_Card"],
        values="Revenue", color="Revenue",
        color_continuous_scale="Blues",
        hover_data={"Customers": True},
    )
    return _apply_theme(fig, "Revenue Treemap — Province & Tier", 500)


# --------------------------------------------------------------------------- #
# Sunburst
# --------------------------------------------------------------------------- #
def education_sunburst(cdf: pd.DataFrame) -> go.Figure:
    grp = (
        cdf.groupby(["Education", "Loyalty_Card"])
           .agg(Revenue=("Total_Revenue", "sum"))
           .reset_index()
    )
    fig = px.sunburst(
        grp, path=["Education", "Loyalty_Card"],
        values="Revenue",
        color_discrete_sequence=QUAL_COLORS,
    )
    return _apply_theme(fig, "Revenue by Education & Tier", 460)


# --------------------------------------------------------------------------- #
# Funnel
# --------------------------------------------------------------------------- #
def loyalty_funnel(kpis: dict) -> go.Figure:
    tc = kpis["tier_counts"]
    tiers = ["Star", "Nova", "Aurora"]
    vals  = [tc.get(t, 0) for t in tiers]
    fig = go.Figure(go.Funnel(
        y=tiers, x=vals,
        textinfo="value+percent initial",
        marker_color=[PALETTE[t] for t in tiers],
    ))
    return _apply_theme(fig, "Loyalty Tier Funnel", 360)


# --------------------------------------------------------------------------- #
# Waterfall
# --------------------------------------------------------------------------- #
def revenue_waterfall(tier_df: pd.DataFrame) -> go.Figure:
    measures = ["relative"] * len(tier_df) + ["total"]
    x_labels = list(tier_df["Tier"]) + ["Total"]
    y_vals   = list(tier_df["Revenue"]) + [None]
    text     = [f"${v/1e6:.1f}M" for v in tier_df["Revenue"]] + [f"${tier_df['Revenue'].sum()/1e6:.1f}M"]

    fig = go.Figure(go.Waterfall(
        measure=measures, x=x_labels, y=y_vals,
        text=text, textposition="outside",
        connector=dict(line=dict(color="rgba(99,179,237,0.3)", width=1, dash="dot")),
        increasing=dict(marker_color="#68d391"),
        totals=dict(marker_color="#63b3ed"),
    ))
    return _apply_theme(fig, "Revenue Waterfall by Tier", 380)


# --------------------------------------------------------------------------- #
# Pareto
# --------------------------------------------------------------------------- #
def pareto_chart(cdf: pd.DataFrame, top_n: int = 20) -> go.Figure:
    grp = (
        cdf.groupby("Province")["Total_Revenue"]
           .sum()
           .sort_values(ascending=False)
           .head(top_n)
           .reset_index()
    )
    grp["cumulative_pct"] = grp["Total_Revenue"].cumsum() / grp["Total_Revenue"].sum() * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=grp["Province"], y=grp["Total_Revenue"],
                         marker_color=QUAL_COLORS[0], name="Revenue"), secondary_y=False)
    fig.add_trace(go.Scatter(x=grp["Province"], y=grp["cumulative_pct"],
                              mode="lines+markers", line=dict(color="#f6ad55", width=2),
                              name="Cumulative %"), secondary_y=True)
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False,
                     gridcolor="rgba(99,179,237,0.08)")
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True,
                     range=[0, 110], gridcolor="rgba(0,0,0,0)")
    return _apply_theme(fig, "Pareto — Revenue by Province", 420)


# --------------------------------------------------------------------------- #
# Sankey
# --------------------------------------------------------------------------- #
def loyalty_sankey(cdf: pd.DataFrame) -> go.Figure:
    # Education → Tier → Status
    edus   = sorted(cdf["Education"].unique())
    tiers  = sorted(cdf["Loyalty_Card"].unique())
    status = ["Active", "Cancelled"]

    nodes  = edus + tiers + status
    idx    = {n: i for i, n in enumerate(nodes)}
    colors = (
        ["#63b3ed"] * len(edus)
        + [PALETTE.get(t, "#9f7aea") for t in tiers]
        + ["#68d391", "#fc8181"]
    )

    srcs, tgts, vals = [], [], []
    for e in edus:
        for t in tiers:
            v = len(cdf[(cdf["Education"] == e) & (cdf["Loyalty_Card"] == t)])
            if v:
                srcs.append(idx[e]); tgts.append(idx[t]); vals.append(v)
    for t in tiers:
        for s in status:
            cancel_val = 1 if s == "Cancelled" else 0
            v = len(cdf[(cdf["Loyalty_Card"] == t) & (cdf["Cancelled"] == cancel_val)])
            if v:
                srcs.append(idx[t]); tgts.append(idx[s]); vals.append(v)

    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="#0a0e1a", width=0.5),
                  label=nodes, color=colors),
        link=dict(source=srcs, target=tgts, value=vals,
                  color="rgba(99,179,237,0.12)"),
    ))
    return _apply_theme(fig, "Customer Flow: Education → Tier → Status", 500)


# --------------------------------------------------------------------------- #
# Scatter Matrix
# --------------------------------------------------------------------------- #
def scatter_matrix(cdf: pd.DataFrame) -> go.Figure:
    cols = ["CLV", "Total_Flights", "Salary", "Engagement"]
    sub  = cdf[cols + ["Loyalty_Card"]].dropna().sample(min(2000, len(cdf)), random_state=7)
    fig  = px.scatter_matrix(
        sub, dimensions=cols, color="Loyalty_Card",
        color_discrete_map=PALETTE, opacity=0.45,
    )
    fig.update_traces(diagonal_visible=False)
    return _apply_theme(fig, "Scatter Matrix — Key Metrics", 520)


# --------------------------------------------------------------------------- #
# ROC Curve (for ML page)
# --------------------------------------------------------------------------- #
def roc_curve_chart(fpr, tpr, auc_val: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"ROC (AUC={auc_val:.3f})",
                             line=dict(color="#63b3ed", width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Random", line=dict(color="#718096", dash="dash")))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return _apply_theme(fig, "ROC Curve — Churn Prediction", 400)


def feature_importance_bar(feat_names, importances) -> go.Figure:
    df_fi = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    df_fi = df_fi.sort_values("Importance", ascending=True).tail(12)
    fig = go.Figure(go.Bar(
        x=df_fi["Importance"], y=df_fi["Feature"], orientation="h",
        marker_color=QUAL_COLORS[0],
    ))
    return _apply_theme(fig, "Feature Importance", 420)
