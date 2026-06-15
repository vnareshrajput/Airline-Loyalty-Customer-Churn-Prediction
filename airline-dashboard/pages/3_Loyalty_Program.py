"""
pages/3_Loyalty_Program.py
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Loyalty Program", page_icon="🏆", layout="wide")
css_path = os.path.join(ROOT, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from utils.data_loader import load_data, get_customer_df
from utils.kpi_calculator import tier_performance, churn_by_tier, engagement_by_education
from utils.charts import (loyalty_funnel, loyalty_sankey, engagement_box,
                           salary_violin, PALETTE, QUAL_COLORS, _apply_theme)

@st.cache_resource
def _load():
    p = os.path.join(ROOT, "master_airline_loyalty.csv")
    df = load_data(p); cdf = get_customer_df(df)
    return df, cdf

df, cdf = _load()

st.markdown("""
<div class='page-title'>
  <h1>🏆 Loyalty Program Analytics</h1>
  <p>Tier performance, points mechanics, retention metrics & engagement analysis</p>
</div>""", unsafe_allow_html=True)

# ── KPI cards ────────────────────────────────────────────────────────────────
tp     = tier_performance(cdf)
tot_pts_acc = cdf["Total_Points"].sum()
tot_pts_red = cdf["Points_Redeemed"].sum()
redemption  = tot_pts_red / tot_pts_acc * 100 if tot_pts_acc else 0
pts_bal     = cdf["Total_Points"].sum() - cdf["Points_Redeemed"].sum()

cols = st.columns(5)
kpi_items = [
    ("🎯", "Loyalty Tiers",        "3 Tiers",            "Aurora · Nova · Star"),
    ("⭐", "Total Points Acc.",    f"{tot_pts_acc/1e9:.2f}B","Lifetime accumulated"),
    ("🔄", "Total Redeemed",       f"{tot_pts_red/1e6:.0f}M","Points redeemed"),
    ("📊", "Redemption Rate",      f"{redemption:.1f}%",    "Overall program"),
    ("💰", "Points Balance",       f"{pts_bal/1e9:.2f}B",   "Unredeemed points"),
]
for col, (icon, lbl, val, sub) in zip(cols, kpi_items):
    with col:
        st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-icon'>{icon}</div>
        <div class='kpi-label'>{lbl}</div>
        <div class='kpi-value'>{val}</div>
        <div class='kpi-delta neutral'>{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── tier performance table + funnel ──────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🏅 Tier Performance Overview</h2></div>""",
            unsafe_allow_html=True)

c1, c2 = st.columns([2, 1])
with c1:
    disp = tp.copy()
    disp["Avg CLV"]       = disp["Avg_CLV"].map("${:,.0f}".format)
    disp["Avg Flights"]   = disp["Avg_Flights"].map("{:.1f}".format)
    disp["Total Revenue"] = disp["Total_Revenue"].map("${:,.0f}".format)
    disp["Avg Engagement"]= disp["Avg_Engagement"].map("{:.2f}".format)
    disp["Churn Rate"]    = (disp["Churn_Rate"]*100).map("{:.1f}%".format)
    st.dataframe(
        disp[["Tier","Customers","Avg CLV","Avg Flights","Total Revenue","Avg Engagement","Churn Rate"]],
        use_container_width=True, hide_index=True,
    )

    # stacked bar: points acc vs redeemed by tier
    pts_data = cdf.groupby("Loyalty_Card").agg(
        Accumulated=("Total_Points","sum"),
        Redeemed=("Points_Redeemed","sum"),
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accumulated", x=pts_data["Loyalty_Card"],
                         y=pts_data["Accumulated"], marker_color="#63b3ed"))
    fig.add_trace(go.Bar(name="Redeemed", x=pts_data["Loyalty_Card"],
                         y=pts_data["Redeemed"], marker_color="#9f7aea"))
    fig.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0",
                      height=320, title="Points Accumulated vs Redeemed by Tier",
                      legend=dict(bgcolor="rgba(13,21,38,0.8)"))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    from utils.kpi_calculator import executive_kpis
    kpis = executive_kpis(df, cdf)
    st.plotly_chart(loyalty_funnel(kpis), use_container_width=True)

    # redemption donut
    fig = go.Figure(go.Pie(
        labels=["Redeemed","Unredeemed"],
        values=[tot_pts_red, pts_bal],
        hole=0.6,
        marker_colors=["#68d391","#4a5568"],
        textinfo="label+percent",
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0",
                      height=280, title="Redemption Breakdown",
                      margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ── churn analysis ───────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>📉 Churn & Retention Analysis</h2></div>""",
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
cd = churn_by_tier(cdf)

with c1:
    fig = go.Figure(go.Bar(
        x=cd["Tier"], y=cd["Churn_Rate"]*100,
        marker_color=["#fc8181","#f6ad55","#68d391"],
        text=[f"{v:.1f}%" for v in cd["Churn_Rate"]*100],
        textposition="outside",
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=340,
                      title="Churn Rate by Tier",
                      yaxis_title="Churn Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Retention matrix (heatmap: tier x marital)
    ret = cdf.groupby(["Loyalty_Card","Marital_Status"])["Cancelled"].mean().reset_index()
    pivot = ret.pivot(index="Loyalty_Card", columns="Marital_Status", values="Cancelled")
    fig = go.Figure(go.Heatmap(
        z=pivot.values*100, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="RdYlGn_r", text=np.round(pivot.values*100,1),
        texttemplate="%{text}%", textfont=dict(size=12),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0",
                      height=340, title="Churn Rate: Tier × Marital Status (%)")
    st.plotly_chart(fig, use_container_width=True)

with c3:
    # Churn by enrollment year
    enr = cdf.groupby("Enrollment_Yr")["Cancelled"].mean().reset_index()
    enr.columns = ["Year","Churn_Rate"]
    fig = px.line(enr, x="Year", y="Churn_Rate", markers=True,
                  title="Churn Rate by Enrollment Year")
    fig.update_traces(line_color="#63b3ed")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=340)
    st.plotly_chart(fig, use_container_width=True)

# ── engagement deep dive ─────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>⚡ Engagement Deep Dive</h2></div>""",
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(engagement_box(cdf), use_container_width=True)

with c2:
    edu_eng = engagement_by_education(cdf)
    fig = px.bar(edu_eng, x="Education", y="Avg_Engagement",
                 color="Avg_Engagement", color_continuous_scale="Blues",
                 text=edu_eng["Avg_Engagement"].round(2),
                 title="Avg Engagement by Education Level")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=380)
    st.plotly_chart(fig, use_container_width=True)

# ── Sankey ───────────────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🔀 Customer Flow — Sankey Diagram</h2>
<p>Education → Loyalty Tier → Status</p></div>""", unsafe_allow_html=True)
st.plotly_chart(loyalty_sankey(cdf), use_container_width=True)

# ── points timeline ──────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>📈 Points Accumulation Timeline</h2></div>""",
            unsafe_allow_html=True)

pts_monthly = df.groupby(["Date","Loyalty Card"]).agg(
    Points=("Points Accumulated","sum"),
    Redeemed=("Points Redeemed","sum"),
).reset_index()

fig = px.line(pts_monthly, x="Date", y="Points", color="Loyalty Card",
              color_discrete_map=PALETTE, title="Monthly Points Accumulated by Tier")
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font_color="#a0aec0", height=380,
                  legend=dict(bgcolor="rgba(13,21,38,0.8)"))
st.plotly_chart(fig, use_container_width=True)

# ── business recommendations ─────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>💡 Loyalty Program Recommendations</h2></div>""",
            unsafe_allow_html=True)

worst_churn_tier = cd.sort_values("Churn_Rate",ascending=False).iloc[0]
best_engage_tier = tp.sort_values("Avg_Engagement",ascending=False).iloc[0]

recs = [
    ("🎯", f"<strong>Retention priority:</strong> <strong>{worst_churn_tier['Tier']}</strong> "
           f"has the highest churn at <strong>{worst_churn_tier['Churn_Rate']*100:.1f}%</strong>. "
           "Implement targeted win-back campaigns with bonus point multipliers."),
    ("🚀", f"<strong>Engagement leader:</strong> <strong>{best_engage_tier['Tier']}</strong> "
           f"scores <strong>{best_engage_tier['Avg_Engagement']:.2f}/10</strong> in engagement. "
           "Replicate successful engagement strategies across other tiers."),
    ("🔄", f"<strong>Redemption nudge:</strong> {100-redemption:.0f}% of points are unredeemed. "
           "Introduce expiry-based alerts and bonus redemption windows to drive engagement."),
    ("⬆", "Consider a <strong>tier upgrade fast-track</strong> for Star members with 15+ flights "
           "per year — this segment has high flight frequency but lower average CLV."),
    ("💳", "Launch a <strong>co-branded credit card</strong> offer targeting Aurora members — "
           "this tier's high CLV and low churn make them ideal for cross-sell products."),
]
for icon, text in recs:
    st.markdown(f"""<div class='insight-card'>
    <div class='insight-icon'>{icon}</div>
    <div class='insight-text'>{text}</div></div>""", unsafe_allow_html=True)
