"""
pages/2_Customer_Analytics.py
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

st.set_page_config(page_title="Customer Analytics", page_icon="👥", layout="wide")
css_path = os.path.join(ROOT, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from utils.data_loader import load_data, get_customer_df
from utils.charts import (clv_histogram, salary_violin, engagement_box,
                           correlation_heatmap, scatter_matrix,
                           province_flight_heatmap, clv_vs_flights_scatter,
                           engagement_vs_redemption, gender_donut, education_sunburst)

@st.cache_resource
def _load():
    p = os.path.join(ROOT, "master_airline_loyalty.csv")
    df = load_data(p); cdf = get_customer_df(df)
    return df, cdf

df, cdf = _load()

st.markdown("""
<div class='page-title'>
  <h1>👥 Customer Analytics</h1>
  <p>Demographics, segmentation, and behavioral analysis with dynamic filters</p>
</div>""", unsafe_allow_html=True)

# ── sidebar filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filters")
    sel_tier    = st.multiselect("Loyalty Tier",    cdf["Loyalty_Card"].unique().tolist(),
                                  default=cdf["Loyalty_Card"].unique().tolist())
    sel_gender  = st.multiselect("Gender",          cdf["Gender"].unique().tolist(),
                                  default=cdf["Gender"].unique().tolist())
    sel_edu     = st.multiselect("Education",       cdf["Education"].unique().tolist(),
                                  default=cdf["Education"].unique().tolist())
    sel_prov    = st.multiselect("Province (top 5 default)",
                                  cdf["Province"].unique().tolist(),
                                  default=cdf["Province"].value_counts().head(5).index.tolist())
    clv_range   = st.slider("CLV Range ($)",
                             int(cdf["CLV"].min()), int(cdf["CLV"].max()),
                             (int(cdf["CLV"].min()), int(cdf["CLV"].max())))
    sal_range   = st.slider("Salary Range ($)",
                             int(cdf["Salary"].min()), int(cdf["Salary"].max()),
                             (int(cdf["Salary"].min()), int(cdf["Salary"].max())))

mask = (
    cdf["Loyalty_Card"].isin(sel_tier)
    & cdf["Gender"].isin(sel_gender)
    & cdf["Education"].isin(sel_edu)
    & cdf["Province"].isin(sel_prov)
    & cdf["CLV"].between(*clv_range)
    & cdf["Salary"].between(*sal_range)
)
fdf = cdf[mask].copy()

st.markdown(f"<p style='color:#718096; font-size:13px;'>Showing <strong style='color:#63b3ed'>"
            f"{len(fdf):,}</strong> of {len(cdf):,} customers after filters</p>",
            unsafe_allow_html=True)

# ── demographics row ─────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🧑‍🤝‍🧑 Demographics</h2></div>""",
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.plotly_chart(gender_donut(fdf), use_container_width=True)

with c2:
    edu_counts = fdf["Education"].value_counts().reset_index()
    edu_counts.columns = ["Education", "Count"]
    fig = px.bar(edu_counts, x="Count", y="Education", orientation="h",
                 color="Count", color_continuous_scale="Blues",
                 title="Customers by Education")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=320,
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

with c3:
    ms = fdf["Marital_Status"].value_counts().reset_index()
    ms.columns = ["Marital Status", "Count"]
    fig = px.pie(ms, names="Marital Status", values="Count",
                 hole=0.45, title="Marital Status",
                 color_discrete_sequence=["#63b3ed","#9f7aea","#68d391","#f6ad55"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0", height=320)
    st.plotly_chart(fig, use_container_width=True)

# ── segmentation ─────────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🗂 Segmentation Analysis</h2></div>""",
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["💰 CLV", "💼 Salary", "⭐ Engagement", "✈ Flights"])

with tab1:
    st.plotly_chart(clv_histogram(fdf), use_container_width=True)
    avg_per_tier = fdf.groupby("Loyalty_Card")["CLV"].mean().reset_index()
    for _, row in avg_per_tier.iterrows():
        st.markdown(f"""<div class='insight-card'>
        <div class='insight-icon'>💳</div>
        <div class='insight-text'><strong>{row['Loyalty_Card']}</strong> tier avg CLV:
        <strong>${row['CLV']:,.0f}</strong></div></div>""", unsafe_allow_html=True)

with tab2:
    st.plotly_chart(salary_violin(fdf), use_container_width=True)

with tab3:
    st.plotly_chart(engagement_box(fdf), use_container_width=True)

with tab4:
    fig = px.histogram(fdf, x="Total_Flights", nbins=50, color="Loyalty_Card",
                       color_discrete_map={"Aurora":"#63b3ed","Nova":"#9f7aea","Star":"#68d391"},
                       barmode="overlay", opacity=0.7,
                       title="Flight Frequency Distribution")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ── geo & heatmaps ───────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🗺 Geographic & Heatmap Analysis</h2></div>""",
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    prov_counts = fdf.groupby("Province")["Loyalty Number"].count().nlargest(12).reset_index()
    prov_counts.columns = ["Province","Customers"]
    fig = px.bar(prov_counts, x="Customers", y="Province", orientation="h",
                 color="Customers", color_continuous_scale="Blues",
                 title="Top Provinces by Customer Count")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=420, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.plotly_chart(province_flight_heatmap(df[df["Province"].isin(sel_prov)]),
                    use_container_width=True)

# ── scatter analysis ─────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🔵 Behavioural Scatter Analysis</h2></div>""",
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(clv_vs_flights_scatter(fdf), use_container_width=True)
with c2:
    st.plotly_chart(engagement_vs_redemption(fdf), use_container_width=True)

# ── correlation matrix ───────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🔗 Correlation Matrix</h2></div>""",
            unsafe_allow_html=True)
st.plotly_chart(correlation_heatmap(fdf), use_container_width=True)

# ── salary group analysis ─────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>💵 Income Group Analysis</h2></div>""",
            unsafe_allow_html=True)

sal_grp = fdf.groupby(["Salary_Group","Loyalty_Card"]).agg(
    Count=("Loyalty Number","count"),
    Avg_CLV=("CLV","mean"),
    Avg_Flights=("Total_Flights","mean"),
).reset_index()

fig = px.bar(sal_grp, x="Salary_Group", y="Count", color="Loyalty_Card",
             barmode="group",
             color_discrete_map={"Aurora":"#63b3ed","Nova":"#9f7aea","Star":"#68d391"},
             title="Customer Count by Income Group & Tier",
             category_orders={"Salary_Group":["< $40K","$40K–$70K","$70K–$100K","$100K–$150K","$150K+"]})
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font_color="#a0aec0", height=380)
st.plotly_chart(fig, use_container_width=True)

# ── sunburst ─────────────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🌐 Education × Tier Revenue Sunburst</h2></div>""",
            unsafe_allow_html=True)
st.plotly_chart(education_sunburst(fdf), use_container_width=True)

# ── auto insights ────────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>💡 Customer Intelligence Insights</h2></div>""",
            unsafe_allow_html=True)

top_edu = fdf.groupby("Education")["CLV"].mean().idxmax()
top_prov = fdf.groupby("Province")["Total_Revenue"].sum().idxmax()
gender_clv = fdf.groupby("Gender")["CLV"].mean()

insights = [
    ("🎓", f"Customers with <strong>{top_edu}</strong> education have the highest average CLV "
           f"at <strong>${fdf[fdf['Education']==top_edu]['CLV'].mean():,.0f}</strong>."),
    ("🗺", f"<strong>{top_prov}</strong> leads all provinces in total revenue contribution "
           f"from this filtered segment."),
    ("👩", f"Female avg CLV: <strong>${gender_clv.get('Female',0):,.0f}</strong> | "
           f"Male avg CLV: <strong>${gender_clv.get('Male',0):,.0f}</strong>."),
    ("📊", f"Salary and CLV show a moderate positive correlation — income-based tier "
           f"upgrade campaigns could improve revenue by targeting $70K–$100K earners."),
]
for icon, text in insights:
    st.markdown(f"""<div class='insight-card'>
    <div class='insight-icon'>{icon}</div>
    <div class='insight-text'>{text}</div></div>""", unsafe_allow_html=True)

# ── download ─────────────────────────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
st.download_button("⬇ Download Filtered Customer Data",
                   fdf.to_csv(index=False).encode(),
                   "filtered_customers.csv", "text/csv")
