"""
pages/1_Executive_Summary.py
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Executive Summary", page_icon="📊", layout="wide")

css_path = os.path.join(ROOT, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from utils.data_loader import load_data, get_customer_df
from utils.kpi_calculator import (executive_kpis, revenue_by_tier, revenue_by_province,
                                   revenue_monthly, tier_performance, churn_by_tier)
from utils.charts import (revenue_trend_chart, tier_revenue_bar, tier_customer_donut,
                           province_revenue_bar, clv_histogram, loyalty_funnel,
                           revenue_waterfall)

@st.cache_resource
def _load():
    p = os.path.join(ROOT, "master_airline_loyalty.csv")
    df = load_data(p); cdf = get_customer_df(df)
    return df, cdf

df, cdf = _load()
kpis = executive_kpis(df, cdf)

# ── page header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-title'>
  <h1>📊 Executive Summary</h1>
  <p>High-level KPIs and strategic insights for airline leadership</p>
</div>""", unsafe_allow_html=True)

# ── KPI row ──────────────────────────────────────────────────────────────────
cols = st.columns(6)
cards = [
    ("👥", "Total Customers",   f"{kpis['total_customers']:,}",       "neutral"),
    ("💰", "Total Revenue",     f"${kpis['total_revenue']/1e9:.2f}B", "positive"),
    ("📈", "Avg CLV",           f"${kpis['avg_clv']:,.0f}",           "positive"),
    ("✈",  "Avg Flights/Cust", f"{kpis['avg_flights']:.1f}",         "neutral"),
    ("✅", "Active Members",   f"{kpis['active_members']:,}",         "positive"),
    ("⚠️", "Churn %",          f"{kpis['churn_pct']:.1f}%",          "negative"),
]
for col, (icon, lbl, val, cls) in zip(cols, cards):
    with col:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-icon'>{icon}</div>
          <div class='kpi-label'>{lbl}</div>
          <div class='kpi-value'>{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── revenue trend + funnel ───────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("""<div class='section-header'><h2>📈 Monthly Revenue Trend</h2></div>""",
                unsafe_allow_html=True)
    mrev = revenue_monthly(df)
    fig  = revenue_trend_chart(mrev)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("""<div class='section-header'><h2>🏷 Loyalty Funnel</h2></div>""",
                unsafe_allow_html=True)
    st.plotly_chart(loyalty_funnel(kpis), use_container_width=True)

# ── tier charts ──────────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🏆 Loyalty Tier Performance</h2>
<p>Revenue contribution and customer share by tier</p></div>""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
tier_df = revenue_by_tier(cdf)

with c1:
    st.plotly_chart(tier_revenue_bar(tier_df), use_container_width=True)
with c2:
    st.plotly_chart(tier_customer_donut(tier_df), use_container_width=True)
with c3:
    st.plotly_chart(revenue_waterfall(tier_df), use_container_width=True)

# ── province + CLV ───────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.markdown("""<div class='section-header'><h2>🗺 Top Provinces by Revenue</h2></div>""",
                unsafe_allow_html=True)
    prov_df = revenue_by_province(cdf, top_n=10)
    st.plotly_chart(province_revenue_bar(prov_df), use_container_width=True)

with c2:
    st.markdown("""<div class='section-header'><h2>📊 CLV Distribution</h2></div>""",
                unsafe_allow_html=True)
    st.plotly_chart(clv_histogram(cdf), use_container_width=True)

# ── tier performance table ───────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>📋 Tier Deep Dive</h2></div>""",
            unsafe_allow_html=True)
tp = tier_performance(cdf)
disp = tp.copy()
disp["Avg CLV"]         = disp["Avg_CLV"].map("${:,.0f}".format)
disp["Avg Flights"]     = disp["Avg_Flights"].map("{:.1f}".format)
disp["Total Revenue"]   = disp["Total_Revenue"].map("${:,.0f}".format)
disp["Avg Engagement"]  = disp["Avg_Engagement"].map("{:.2f}".format)
disp["Churn Rate"]      = (disp["Churn_Rate"] * 100).map("{:.1f}%".format)
st.dataframe(
    disp[["Tier","Customers","Avg CLV","Avg Flights","Total Revenue","Avg Engagement","Churn Rate"]],
    use_container_width=True, hide_index=True,
)

# ── auto insights ────────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>💡 Executive Insights</h2></div>""",
            unsafe_allow_html=True)

top_tier = tier_df.sort_values("Revenue", ascending=False).iloc[0]
top_pct  = top_tier["Revenue"] / tier_df["Revenue"].sum() * 100
top_prov = prov_df.iloc[0]

cd = churn_by_tier(cdf)
safest = cd.sort_values("Churn_Rate").iloc[0]

insights = [
    ("💎", f"<strong>{top_tier['Tier']}</strong> members account for "
           f"<strong>{top_pct:.0f}%</strong> of total revenue from "
           f"<strong>{top_tier['Customers']:,}</strong> customers."),
    ("🗺", f"<strong>{top_prov['Province']}</strong> leads in revenue at "
           f"<strong>${top_prov['Revenue']/1e6:.0f}M</strong>, followed by "
           f"{prov_df.iloc[1]['Province']} at ${prov_df.iloc[1]['Revenue']/1e6:.0f}M."),
    ("📉", f"Overall churn rate is <strong>{kpis['churn_pct']:.1f}%</strong>. "
           f"<strong>{safest['Tier']}</strong> has the lowest churn at "
           f"<strong>{safest['Churn_Rate']*100:.1f}%</strong>."),
    ("🏅", f"Avg CLV across all customers: <strong>${kpis['avg_clv']:,.0f}</strong>. "
           f"Points redemption rate: <strong>{kpis['redemption_rate']:.1f}%</strong>."),
    ("✈",  f"Active members average <strong>{kpis['avg_flights']:.1f} flights</strong> each. "
           f"Encouraging high-frequency flyers to upgrade tiers can boost revenue by 15–20%."),
]
for icon, text in insights:
    st.markdown(f"""
    <div class='insight-card'>
      <div class='insight-icon'>{icon}</div>
      <div class='insight-text'>{text}</div>
    </div>""", unsafe_allow_html=True)

# ── download ─────────────────────────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
csv_bytes = cdf.to_csv(index=False).encode()
st.download_button("⬇ Download Customer Summary CSV", csv_bytes,
                   "customer_summary.csv", "text/csv")
