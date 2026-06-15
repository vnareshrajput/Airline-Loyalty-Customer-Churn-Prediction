"""
app.py — Airline Customer Loyalty Analytics Dashboard
Main entry point / Home page
"""
import streamlit as st
import os, sys

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Airline Loyalty Analytics",
    page_icon   = "✈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── inject CSS ───────────────────────────────────────────────────────────────
css_path = os.path.join(ROOT, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── data bootstrap ───────────────────────────────────────────────────────────
from utils.data_loader import load_data, get_customer_df

@st.cache_resource
def bootstrap():
    default_csv = os.path.join(ROOT, "master_airline_loyalty.csv")
    df  = load_data(default_csv)
    cdf = get_customer_df(df)
    return df, cdf

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px 0;'>
      <div style='font-size:36px;'>✈</div>
      <div style='font-size:17px; font-weight:800; color:#e2e8f0; letter-spacing:.5px;'>
        Airline Loyalty
      </div>
      <div style='font-size:11px; color:#718096; letter-spacing:1px; text-transform:uppercase;'>
        Analytics Platform
      </div>
    </div>
    <hr style='border-color:rgba(99,179,237,0.15); margin:12px 0;'/>
    """, unsafe_allow_html=True)

    # ── optional CSV upload ──────────────────────────────────────────────────
    # uploaded = st.file_uploader("Upload Custom CSV", type=["csv"],
    #                              help="Override the default dataset")
    # if uploaded:
    #     import io, pandas as pd
    #     try:
    #         raw = pd.read_csv(io.BytesIO(uploaded.read()))
    #         raw.to_csv(os.path.join(ROOT, "master_airline_loyalty.csv"), index=False)
    #         st.cache_data.clear()
    #         st.success("Dataset replaced. Reload pages to apply.")
    #     except Exception as exc:
    #         st.error(f"Could not parse: {exc}")

    # st.markdown("""
    # <hr style='border-color:rgba(99,179,237,0.1); margin:12px 0;'/>
    # <div style='font-size:11px; color:#4a5568; text-align:center; padding-bottom:8px;'>
    #   Powered by Streamlit · Plotly · Scikit-Learn
    # </div>
    # """, unsafe_allow_html=True)

# ── hero section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-title'>
  <h1>✈ Airline Customer Loyalty Analytics</h1>
  <p>Fortune-500-grade intelligence platform — Executive Edition</p>
</div>
""", unsafe_allow_html=True)

# load data
try:
    df, cdf = bootstrap()
except Exception as exc:
    st.error(f"Data load failed: {exc}")
    st.stop()

from utils.kpi_calculator import executive_kpis
kpis = executive_kpis(df, cdf)

# ── top KPI row ──────────────────────────────────────────────────────────────
cols = st.columns(6)
kpi_data = [
    ("👥", "Total Customers",  f"{kpis['total_customers']:,}",      "All enrolled members", "neutral"),
    ("💰", "Total Revenue",    f"${kpis['total_revenue']/1e9:.2f}B","Estimated lifetime",   "positive"),
    ("📈", "Avg CLV",          f"${kpis['avg_clv']:,.0f}",          "Per customer",         "positive"),
    ("✈",  "Avg Flights",      f"{kpis['avg_flights']:.1f}",        "Per customer",         "neutral"),
    ("✅", "Active Members",   f"{kpis['active_members']:,}",       f"{100-kpis['churn_pct']:.1f}% of total", "positive"),
    ("⚠️", "Churn Risk %",     f"{kpis['churn_pct']:.1f}%",        "Cancelled segment",    "negative"),
]
for col, (icon, label, val, sub, cls) in zip(cols, kpi_data):
    with col:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-icon'>{icon}</div>
          <div class='kpi-label'>{label}</div>
          <div class='kpi-value'>{val}</div>
          <div class='kpi-delta {cls}'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── secondary metrics ─────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""<div class='section-header'><h2>🗺 Navigation Guide</h2>
    <p>Use the sidebar to explore each analytics page</p></div>""", unsafe_allow_html=True)
    for pg, desc in [
        ("1 · Executive Summary",   "Top-level KPIs, revenue trends & tier overview"),
        ("2 · Customer Analytics",  "Demographics, segmentation & behavioral filters"),
        ("3 · Loyalty Program",     "Tier performance, points & retention analysis"),
        ("4 · Revenue Insights",    "Waterfall, Pareto, Treemap & decomposition"),
        ("5 · Predictive Analytics","ML churn model, CLV prediction & risk scoring"),
    ]:
        st.markdown(f"""
        <div class='insight-card'>
          <div class='insight-icon'>📌</div>
          <div class='insight-text'><strong>{pg}</strong><br>{desc}</div>
        </div>""", unsafe_allow_html=True)

with c2:
    st.markdown("""<div class='section-header'><h2>📊 Dataset Snapshot</h2>
    <p>Key statistics about the loaded data</p></div>""", unsafe_allow_html=True)
    snap = [
        ("Records",      f"{len(df):,}"),
        ("Unique Customers", f"{cdf['Loyalty Number'].nunique():,}"),
        ("Loyalty Tiers",    "Aurora · Nova · Star"),
        ("Provinces",        f"{df['Province'].nunique()}"),
        ("Date Range",       f"{df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}"),
        ("Redemption Rate",  f"{kpis['redemption_rate']:.1f}%"),
        ("Avg Engagement",   f"{kpis['avg_engagement']:.2f} / 10"),
    ]
    for label, val in snap:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; padding:9px 14px;
        border-bottom:1px solid rgba(99,179,237,0.08); font-size:13px;'>
          <span style='color:#718096;'>{label}</span>
          <span style='color:#e2e8f0; font-weight:600;'>{val}</span>
        </div>""", unsafe_allow_html=True)

with c3:
    st.markdown("""<div class='section-header'><h2>💡 Executive Highlights</h2>
    <p>Auto-generated observations from the data</p></div>""", unsafe_allow_html=True)

    from utils.kpi_calculator import revenue_by_tier
    trev = revenue_by_tier(cdf)
    top_tier  = trev.sort_values("Revenue", ascending=False).iloc[0]
    top_pct   = top_tier["Revenue"] / trev["Revenue"].sum() * 100
    top_cust_pct = top_tier["Customers"] / trev["Customers"].sum() * 100

    insights = [
        ("💎", f"<strong>{top_tier['Tier']}</strong> members drive "
               f"<strong>{top_pct:.0f}%</strong> of total revenue while representing "
               f"only <strong>{top_cust_pct:.0f}%</strong> of customers."),
        ("📉", f"Churn rate stands at <strong>{kpis['churn_pct']:.1f}%</strong>. "
               "Targeted retention could recover significant CLV."),
        ("🎯", f"Average engagement score is <strong>{kpis['avg_engagement']:.2f}/10</strong>. "
               "High-engagement customers show 2× higher CLV."),
        ("🏆", f"Points redemption rate is <strong>{kpis['redemption_rate']:.1f}%</strong>, "
               "indicating strong program participation."),
    ]
    for icon, text in insights:
        st.markdown(f"""
        <div class='insight-card'>
          <div class='insight-icon'>{icon}</div>
          <div class='insight-text'>{text}</div>
        </div>""", unsafe_allow_html=True)

# ── tier composition table ───────────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("""<div class='section-header'>
<h2>🏷 Loyalty Tier Composition</h2>
<p>Comparative breakdown across all three loyalty tiers</p>
</div>""", unsafe_allow_html=True)

from utils.kpi_calculator import tier_performance
tp = tier_performance(cdf)
tp["Avg CLV"]         = tp["Avg_CLV"].map("${:,.0f}".format)
tp["Avg Flights"]     = tp["Avg_Flights"].map("{:.1f}".format)
tp["Total Revenue"]   = tp["Total_Revenue"].map("${:,.0f}".format)
tp["Avg Engagement"]  = tp["Avg_Engagement"].map("{:.2f}".format)
tp["Churn Rate"]      = (tp["Churn_Rate"] * 100).map("{:.1f}%".format)
st.dataframe(
    tp[["Tier","Customers","Avg CLV","Avg Flights","Total Revenue","Avg Engagement","Churn Rate"]],
    use_container_width=True, hide_index=True,
)
