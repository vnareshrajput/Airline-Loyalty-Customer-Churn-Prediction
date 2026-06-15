"""
pages/4_Revenue_Insights.py
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Revenue Insights", page_icon="💰", layout="wide")
css_path = os.path.join(ROOT, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from utils.data_loader import load_data, get_customer_df
from utils.kpi_calculator import (revenue_by_tier, revenue_by_province,
                                   revenue_monthly, tier_performance)
from utils.charts import (revenue_trend_chart, tier_revenue_bar, province_revenue_bar,
                           revenue_waterfall, province_treemap, pareto_chart,
                           education_sunburst, PALETTE, QUAL_COLORS)

@st.cache_resource
def _load():
    p = os.path.join(ROOT, "master_airline_loyalty.csv")
    df = load_data(p); cdf = get_customer_df(df)
    return df, cdf

df, cdf = _load()
mrev    = revenue_monthly(df)
tier_df = revenue_by_tier(cdf)
prov_df = revenue_by_province(cdf, top_n=12)
tp      = tier_performance(cdf)

st.markdown("""
<div class='page-title'>
  <h1>💰 Revenue Insights</h1>
  <p>Deep revenue decomposition, geographic analysis, and executive findings</p>
</div>""", unsafe_allow_html=True)

# ── KPI row ──────────────────────────────────────────────────────────────────
total_rev     = cdf["Total_Revenue"].sum()
avg_rev_cust  = cdf["Total_Revenue"].mean()
top_prov_rev  = prov_df.iloc[0]["Revenue"] if len(prov_df) else 0
top_tier_rev  = tier_df.sort_values("Revenue",ascending=False).iloc[0]["Revenue"]
pts_cost      = cdf["Points_Redeemed"].sum() * 0.01  # $0.01/pt est

cols = st.columns(5)
items = [
    ("💰","Total Revenue",      f"${total_rev/1e9:.2f}B"),
    ("👤","Rev / Customer",     f"${avg_rev_cust:,.0f}"),
    ("🏆","Top Tier Rev",       f"${top_tier_rev/1e6:.0f}M"),
    ("🗺","Top Province Rev",   f"${top_prov_rev/1e6:.0f}M"),
    ("🎫","Points Cost Est.",   f"${pts_cost/1e6:.0f}M"),
]
for col,(icon,lbl,val) in zip(cols,items):
    with col:
        st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-icon'>{icon}</div>
        <div class='kpi-label'>{lbl}</div>
        <div class='kpi-value'>{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── revenue trend ─────────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>📈 Revenue Trend Analysis</h2></div>""",
            unsafe_allow_html=True)

c1, c2 = st.columns([3,1])
with c1:
    fig = revenue_trend_chart(mrev)
    # add rolling average
    mrev_copy = mrev.copy()
    mrev_copy["MA3"] = mrev_copy["Revenue"].rolling(3).mean()
    fig.add_trace(go.Scatter(
        x=mrev_copy["Date"], y=mrev_copy["MA3"],
        mode="lines", name="3-Mo MA",
        line=dict(color="#f6ad55", width=2, dash="dot"),
    ))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    yrev = df.groupby("Year")["Revenue"].sum().reset_index()
    yrev["YoY"] = yrev["Revenue"].pct_change()*100
    fig = go.Figure(go.Bar(
        x=yrev["Year"].astype(str), y=yrev["Revenue"],
        marker_color=QUAL_COLORS[0],
        text=[f"${v/1e6:.0f}M" for v in yrev["Revenue"]],
        textposition="outside",
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=350, title="Annual Revenue",
                      margin=dict(l=20,r=10,t=45,b=30))
    st.plotly_chart(fig, use_container_width=True)

# ── tier revenue ──────────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🏷 Revenue by Loyalty Tier</h2></div>""",
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: st.plotly_chart(tier_revenue_bar(tier_df), use_container_width=True)
with c2: st.plotly_chart(revenue_waterfall(tier_df), use_container_width=True)
with c3:
    # Revenue per customer by tier
    rpc = tp.copy()
    rpc["Rev_Per_Customer"] = rpc["Total_Revenue"] / rpc["Customers"]
    fig = go.Figure(go.Bar(
        x=rpc["Tier"], y=rpc["Rev_Per_Customer"],
        marker_color=[PALETTE.get(t,"#63b3ed") for t in rpc["Tier"]],
        text=[f"${v:,.0f}" for v in rpc["Rev_Per_Customer"]],
        textposition="outside",
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=350, title="Revenue per Customer by Tier",
                      margin=dict(l=20,r=20,t=45,b=30))
    st.plotly_chart(fig, use_container_width=True)

# ── geographic analysis ───────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🗺 Geographic Revenue Analysis</h2></div>""",
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1: st.plotly_chart(province_revenue_bar(prov_df), use_container_width=True)
with c2: st.plotly_chart(pareto_chart(cdf), use_container_width=True)

# ── treemap ──────────────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🌳 Revenue Treemap — Province & Tier</h2></div>""",
            unsafe_allow_html=True)
st.plotly_chart(province_treemap(cdf), use_container_width=True)

# ── sunburst ─────────────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🌐 Revenue Sunburst — Education & Tier</h2></div>""",
            unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(education_sunburst(cdf), use_container_width=True)

with c2:
    # Revenue by Marital Status × Tier
    ms_rev = cdf.groupby(["Marital_Status","Loyalty_Card"])["Total_Revenue"].sum().reset_index()
    fig = px.sunburst(ms_rev, path=["Marital_Status","Loyalty_Card"], values="Total_Revenue",
                      color_discrete_sequence=QUAL_COLORS,
                      title="Revenue by Marital Status & Tier")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0", height=460,
                      margin=dict(l=10,r=10,t=45,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ── revenue decomposition heatmap ─────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🔥 Revenue Heatmap — Province × Tier</h2></div>""",
            unsafe_allow_html=True)

heat = cdf.groupby(["Province","Loyalty_Card"])["Total_Revenue"].sum().unstack(fill_value=0)
heat = heat.loc[heat.sum(axis=1).nlargest(12).index]
fig = go.Figure(go.Heatmap(
    z=heat.values/1e6, x=heat.columns.tolist(), y=heat.index.tolist(),
    colorscale="Blues", text=np.round(heat.values/1e6,1),
    texttemplate="$%{text}M", textfont=dict(size=10),
    colorbar=dict(title="Revenue (M$)"),
))
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0",
                  height=480, title="Revenue ($M) by Province & Tier",
                  margin=dict(l=120,r=20,t=45,b=30))
st.plotly_chart(fig, use_container_width=True)

# ── enrollment type revenue ───────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>📋 Revenue by Enrollment Vintage</h2></div>""",
            unsafe_allow_html=True)

enr = cdf.groupby("Enrollment_Yr").agg(
    Revenue=("Total_Revenue","sum"),
    Customers=("Loyalty Number","count"),
    Avg_CLV=("CLV","mean"),
).reset_index()

fig = make_subplots(specs=[[{"secondary_y":True}]])
fig.add_trace(go.Bar(x=enr["Enrollment_Yr"].astype(str), y=enr["Revenue"],
                     name="Revenue", marker_color=QUAL_COLORS[0]), secondary_y=False)
fig.add_trace(go.Scatter(x=enr["Enrollment_Yr"].astype(str), y=enr["Avg_CLV"],
                          name="Avg CLV", line=dict(color="#f6ad55",width=2), mode="lines+markers"),
              secondary_y=True)
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font_color="#a0aec0", height=380, title="Revenue & Avg CLV by Enrollment Year",
                  legend=dict(bgcolor="rgba(13,21,38,0.8)"))
fig.update_yaxes(title_text="Revenue ($)", secondary_y=False, gridcolor="rgba(99,179,237,0.08)")
fig.update_yaxes(title_text="Avg CLV ($)", secondary_y=True, gridcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

# ── executive findings ────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>💡 Revenue Executive Findings</h2></div>""",
            unsafe_allow_html=True)

top_tier = tier_df.sort_values("Revenue",ascending=False).iloc[0]
pct_rev  = top_tier["Revenue"]/total_rev*100
pct_cust = top_tier["Customers"]/tier_df["Customers"].sum()*100
top_p    = prov_df.iloc[0]

# Pareto insight: top 3 provinces
top3_prov = prov_df.head(3)
top3_pct  = top3_prov["Revenue"].sum() / cdf["Total_Revenue"].sum() * 100

findings = [
    ("💎", f"<strong>{top_tier['Tier']}</strong> generates <strong>{pct_rev:.0f}%</strong> of "
           f"total revenue with only <strong>{pct_cust:.0f}%</strong> of customers — "
           f"a revenue concentration ratio of <strong>{pct_rev/pct_cust:.1f}×</strong>."),
    ("🗺", f"Top 3 provinces (<strong>{', '.join(top3_prov['Province'].tolist())}</strong>) "
           f"account for <strong>{top3_pct:.0f}%</strong> of total revenue, "
           "confirming a Pareto-like geographic concentration."),
    ("📅", f"The highest revenue year was <strong>{yrev.loc[yrev['Revenue'].idxmax(),'Year']}</strong>. "
           "Year-over-year trend reflects fleet expansion and loyalty maturation."),
    ("🧮", f"Revenue per customer varies significantly across tiers — "
           f"from ${tp['Total_Revenue'].min()/tp['Customers'].min():,.0f} to "
           f"${tp['Total_Revenue'].max()/tp['Customers'].max():,.0f} — "
           "suggesting high return on tier upgrade investments."),
    ("🎫", f"Estimated loyalty redemption cost of <strong>${pts_cost/1e6:.0f}M</strong> "
           "should be weighed against incremental revenue from member retention."),
]
for icon, text in findings:
    st.markdown(f"""<div class='insight-card'>
    <div class='insight-icon'>{icon}</div>
    <div class='insight-text'>{text}</div></div>""", unsafe_allow_html=True)

# ── download ─────────────────────────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
rev_summary = cdf.groupby(["Province","Loyalty_Card"]).agg(
    Revenue=("Total_Revenue","sum"), Customers=("Loyalty Number","count")
).reset_index()
st.download_button("⬇ Download Revenue Summary CSV",
                   rev_summary.to_csv(index=False).encode(),
                   "revenue_summary.csv","text/csv")
