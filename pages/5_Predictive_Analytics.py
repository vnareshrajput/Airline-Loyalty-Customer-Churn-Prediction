"""
pages/5_Predictive_Analytics.py
Churn Prediction — powered by trained RandomForest model
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Predictive Analytics", page_icon="🤖", layout="wide")
css_path = os.path.join(ROOT, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from utils.data_loader import load_data, get_customer_df
from utils.charts import roc_curve_chart, feature_importance_bar, QUAL_COLORS, PALETTE

# ── load dashboard data ────────────────────────────────────────────────────────
@st.cache_resource
def _load():
    p = os.path.join(ROOT, "master_airline_loyalty.csv")
    df = load_data(p)
    cdf = get_customer_df(df)
    return df, cdf

df, cdf = _load()

# ── load trained model artifacts ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading churn model…")
def load_model_artifacts():
    import joblib
    models_dir = os.path.join(ROOT, "models")
    model       = joblib.load(os.path.join(models_dir, "churn_model.pkl"))
    pipeline    = joblib.load(os.path.join(models_dir, "preprocessing_pipeline.pkl"))
    feat_info   = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))
    metrics     = joblib.load(os.path.join(models_dir, "model_metrics.pkl"))
    feat_names  = joblib.load(os.path.join(models_dir, "feature_names.pkl"))
    return model, pipeline, feat_info, metrics, feat_names

model, pipeline, feat_info, saved_metrics, feature_names = load_model_artifacts()
num_cols  = feat_info["num_cols"]
cat_cols  = feat_info["cat_cols"]
bool_cols = feat_info["bool_cols"]

# ── load & prepare segmented data for predictions ─────────────────────────────
@st.cache_data(show_spinner="Preparing prediction dataset…")
def load_segmented():
    seg_path = os.path.join(ROOT, "airline_segmented.csv")
    seg = pd.read_csv(seg_path)
    return seg

@st.cache_data(show_spinner="Scoring all customers…")
def score_all_customers():
    seg = load_segmented()

    # airline_segmented.csv and master_airline_loyalty.csv share the same row
    # order (verified: same Year/Month alignment, same 389 065 rows).
    # We pull Loyalty Number from master to attach to each scored row.
    master = pd.read_csv(os.path.join(ROOT, "master_airline_loyalty.csv"))

    # Build customer-level summary from master for the results card
    cust_master = master.groupby("Loyalty Number").agg(
        CLV=("CLV", "mean"),
        Total_Flights=("Total Flights", "sum"),
        Total_Revenue=("CLV", "mean"),
        Loyalty_Card=("Loyalty Card", "first"),
        Province=("Province", "first"),
        Gender=("Gender", "first"),
        Cancelled=("Cancelled", "max"),
    ).reset_index()

    # Prepare X — convert bool columns to int, drop target & id
    seg_copy = seg.copy()
    for c in bool_cols:
        if c in seg_copy.columns:
            seg_copy[c] = seg_copy[c].astype(int)

    X_raw = seg_copy.drop(columns=["Cancelled", "Loyalty Number"], errors="ignore")

    # Score every row
    X_prep = pipeline.transform(X_raw)
    probs  = model.predict_proba(X_prep)[:, 1]

    # Attach loyalty numbers (index-aligned with master)
    result = pd.DataFrame({
        "Loyalty Number":    master["Loyalty Number"].values,
        "Churn_Probability": np.round(probs, 4),
    })

    # Per-customer: take maximum churn probability across their rows
    agg = result.groupby("Loyalty Number")["Churn_Probability"].max().reset_index()

    # Merge customer profile info
    scored = agg.merge(cust_master, on="Loyalty Number", how="left")

    def risk_label(p):
        if p < 0.40:   return "Low Risk"
        elif p < 0.70: return "Medium Risk"
        else:          return "High Risk"

    scored["Risk_Category"] = scored["Churn_Probability"].apply(risk_label)
    scored["Churn_Pct"]     = (scored["Churn_Probability"] * 100).round(1)
    return scored

# ── page header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-title'>
  <h1>🤖 Predictive Analytics</h1>
  <p>Churn prediction powered by trained Random Forest model • ROC-AUC: 0.986</p>
</div>""", unsafe_allow_html=True)

# ── model performance metrics ──────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>📊 Model Performance</h2>
<p>Metrics from trained RandomForest (n_estimators=500) with SMOTE balancing</p></div>""",
            unsafe_allow_html=True)

metric_cols = st.columns(5)
metric_data = [
    ("🎯", "ROC AUC",   f"{saved_metrics['ROC_AUC']:.3f}",  "positive"),
    ("✅", "Accuracy",   f"{saved_metrics['Accuracy']*100:.1f}%", "positive"),
    ("⚡", "Precision",  f"{saved_metrics['Precision']:.3f}", "neutral"),
    ("🔄", "Recall",     f"{saved_metrics['Recall']:.3f}",    "neutral"),
    ("📈", "F1 Score",   f"{saved_metrics['F1']:.3f}",        "neutral"),
]
for col, (icon, lbl, val, cls) in zip(metric_cols, metric_data):
    with col:
        st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-icon'>{icon}</div>
        <div class='kpi-label'>{lbl}</div>
        <div class='kpi-value'>{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── feature importance ────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🔑 Top Feature Importances</h2>
<p>Features that most influence churn prediction</p></div>""", unsafe_allow_html=True)

fi = model.feature_importances_
fi_df = pd.DataFrame({"Feature": feature_names, "Importance": fi})
fi_df["Feature"] = fi_df["Feature"].str.replace("num__", "").str.replace("cat__", "").str.replace("_", " ").str.title()
fi_top = fi_df.nlargest(15, "Importance").sort_values("Importance")

fig_fi = go.Figure(go.Bar(
    x=fi_top["Importance"], y=fi_top["Feature"],
    orientation="h",
    marker=dict(
        color=fi_top["Importance"],
        colorscale=[[0, "#2b6cb0"], [0.5, "#63b3ed"], [1, "#90cdf4"]],
        showscale=False,
    ),
    text=fi_top["Importance"].map("{:.3f}".format),
    textposition="outside",
))
fig_fi.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#a0aec0", height=420,
    xaxis_title="Importance", yaxis_title="",
    margin=dict(l=20, r=60, t=20, b=40),
    xaxis=dict(showgrid=True, gridcolor="rgba(99,179,237,0.1)"),
)
st.plotly_chart(fig_fi, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── individual customer churn prediction ──────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🔍 Individual Customer Churn Predictor</h2>
<p>Select a customer to view their personalized churn risk assessment</p></div>""",
            unsafe_allow_html=True)

scored = score_all_customers()

# Customer selector
loyalty_numbers = sorted(scored["Loyalty Number"].dropna().astype(str).tolist())
selected_id = st.selectbox(
    "Select Customer by Loyalty Number",
    options=loyalty_numbers,
    index=0,
    help="Search by Loyalty Number to view individual churn prediction"
)

# Get selected customer
cust = scored[scored["Loyalty Number"].astype(str) == selected_id]

if not cust.empty:
    cust_row = cust.iloc[0]
    prob = float(cust_row["Churn_Probability"])
    prob_pct = prob * 100

    # Risk classification
    if prob < 0.40:
        risk_level = "Low Risk"
        risk_color = "#68d391"
        risk_emoji = "🟢"
        prediction = "✅ Likely to Stay"
        pred_color = "#68d391"
    elif prob < 0.70:
        risk_level = "Medium Risk"
        risk_color = "#f6ad55"
        risk_emoji = "🟠"
        prediction = "⚠️ At Risk"
        pred_color = "#f6ad55"
    else:
        risk_level = "High Risk"
        risk_color = "#fc8181"
        risk_emoji = "🔴"
        prediction = "❌ Likely to Churn"
        pred_color = "#fc8181"

    # Layout: gauge + summary card
    col_gauge, col_summary = st.columns([1, 1])

    with col_gauge:
        # Plotly gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_pct,
            number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 36}},
            title={"text": "Churn Probability", "font": {"color": "#a0aec0", "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#718096", "tickfont": {"color": "#a0aec0"}},
                "bar": {"color": risk_color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(99,179,237,0.2)",
                "steps": [
                    {"range": [0, 40],  "color": "rgba(104,211,145,0.15)"},
                    {"range": [40, 70], "color": "rgba(246,173,85,0.15)"},
                    {"range": [70, 100],"color": "rgba(252,129,129,0.15)"},
                ],
                "threshold": {
                    "line": {"color": risk_color, "width": 3},
                    "thickness": 0.75,
                    "value": prob_pct,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0",
            height=300,
            margin=dict(l=30, r=30, t=50, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Risk badge
        st.markdown(f"""
        <div style='text-align:center; margin-top:-10px;'>
          <span style='display:inline-block; padding:10px 28px; border-radius:50px;
            background:rgba(0,0,0,0.3); border:2px solid {risk_color};
            color:{risk_color}; font-weight:800; font-size:16px; letter-spacing:1px;'>
            {risk_emoji} {risk_level}
          </span>
        </div>
        <div style='text-align:center; margin-top:14px;'>
          <span style='font-size:18px; font-weight:700; color:{pred_color};'>
            {prediction}
          </span>
        </div>
        """, unsafe_allow_html=True)

    with col_summary:
        clv_val = cust_row.get("CLV", 0)
        flights_val = cust_row.get("Total_Flights", 0)
        rev_val = cust_row.get("Total_Revenue", 0)
        card_val = cust_row.get("Loyalty_Card", "N/A")
        gender_val = cust_row.get("Gender", "N/A")
        province_val = cust_row.get("Province", "N/A")

        st.markdown(f"""
        <div class='insight-card' style='border-color:{risk_color}; margin-top:0;'>
          <div style='width:100%;'>
            <div style='font-size:14px; font-weight:700; color:#e2e8f0; margin-bottom:16px; letter-spacing:.5px;'>
              📋 Customer Risk Summary
            </div>
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:12px;'>
              <div>
                <div style='font-size:10px; color:#718096; text-transform:uppercase; letter-spacing:1px;'>Loyalty Number</div>
                <div style='font-size:14px; color:#e2e8f0; font-weight:600;'>{selected_id}</div>
              </div>
              <div>
                <div style='font-size:10px; color:#718096; text-transform:uppercase; letter-spacing:1px;'>Loyalty Card</div>
                <div style='font-size:14px; color:#63b3ed; font-weight:600;'>{card_val}</div>
              </div>
              <div>
                <div style='font-size:10px; color:#718096; text-transform:uppercase; letter-spacing:1px;'>CLV</div>
                <div style='font-size:14px; color:#68d391; font-weight:600;'>${clv_val:,.0f}</div>
              </div>
              <div>
                <div style='font-size:10px; color:#718096; text-transform:uppercase; letter-spacing:1px;'>Total Flights</div>
                <div style='font-size:14px; color:#e2e8f0; font-weight:600;'>{flights_val:,.0f}</div>
              </div>
              <div>
                <div style='font-size:10px; color:#718096; text-transform:uppercase; letter-spacing:1px;'>Revenue</div>
                <div style='font-size:14px; color:#68d391; font-weight:600;'>${rev_val:,.0f}</div>
              </div>
              <div>
                <div style='font-size:10px; color:#718096; text-transform:uppercase; letter-spacing:1px;'>Province</div>
                <div style='font-size:14px; color:#e2e8f0; font-weight:600;'>{province_val}</div>
              </div>
              <div>
                <div style='font-size:10px; color:#718096; text-transform:uppercase; letter-spacing:1px;'>Churn Probability</div>
                <div style='font-size:22px; color:{risk_color}; font-weight:800;'>{prob_pct:.1f}%</div>
              </div>
              <div>
                <div style='font-size:10px; color:#718096; text-transform:uppercase; letter-spacing:1px;'>Risk Level</div>
                <div style='font-size:14px; color:{risk_color}; font-weight:700;'>{risk_emoji} {risk_level}</div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Auto-generated insight
        if prob_pct >= 70:
            insight_text = (
                f"Customer <strong>{selected_id}</strong> has a <strong style='color:#fc8181'>{prob_pct:.1f}%</strong> "
                f"churn probability. This is a high-value {card_val} member at critical risk. "
                f"<strong>Recommend:</strong> Immediate personalised retention offer — bonus points, "
                f"tier upgrade incentive, or exclusive loyalty benefit."
            )
        elif prob_pct >= 40:
            insight_text = (
                f"Customer <strong>{selected_id}</strong> shows moderate churn risk at "
                f"<strong style='color:#f6ad55'>{prob_pct:.1f}%</strong>. Engagement is declining. "
                f"<strong>Recommend:</strong> Targeted re-engagement campaign with flight discount "
                f"or double-points promotion to boost activity."
            )
        else:
            insight_text = (
                f"Customer <strong>{selected_id}</strong> is a loyal {card_val} member with only "
                f"<strong style='color:#68d391'>{prob_pct:.1f}%</strong> churn risk. "
                f"<strong>Recommend:</strong> Maintain engagement through regular rewards communications "
                f"and tier-status updates."
            )

        st.markdown(f"""
        <div class='insight-card' style='margin-top:16px;'>
          <div class='insight-icon'>💡</div>
          <div class='insight-text'>{insight_text}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── risk segmentation overview ─────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>⚠️ Risk Segmentation — All Customers</h2>
<p>Population-level churn risk distribution</p></div>""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    risk_counts = scored["Risk_Category"].value_counts().reset_index()
    risk_counts.columns = ["Category", "Count"]
    order  = ["Low Risk", "Medium Risk", "High Risk"]
    colors = ["#68d391", "#f6ad55", "#fc8181"]
    risk_counts = risk_counts.set_index("Category").reindex(order).fillna(0).reset_index()
    fig_risk = go.Figure(go.Bar(
        x=risk_counts["Category"], y=risk_counts["Count"],
        marker_color=colors,
        text=risk_counts["Count"].astype(int),
        textposition="outside",
    ))
    fig_risk.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#a0aec0", height=360,
        title="Customers by Risk Category",
        margin=dict(l=20, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with c2:
    fig_hist = px.histogram(
        scored, x="Churn_Pct", nbins=50,
        color_discrete_sequence=[QUAL_COLORS[0]],
        title="Churn Probability Distribution (%)",
        labels={"Churn_Pct": "Churn Probability (%)"},
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#a0aec0", height=360,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with c3:
    if "Loyalty_Card" in scored.columns:
        risk_tier = scored.groupby(["Loyalty_Card", "Risk_Category"])["Loyalty Number"].count().reset_index()
        risk_tier.columns = ["Tier", "Risk", "Count"]
        fig_tier = px.bar(
            risk_tier, x="Tier", y="Count", color="Risk",
            color_discrete_map={"Low Risk": "#68d391", "Medium Risk": "#f6ad55", "High Risk": "#fc8181"},
            title="Risk Category by Loyalty Tier",
            category_orders={"Risk": order},
        )
        fig_tier.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", height=360,
            legend=dict(bgcolor="rgba(13,21,38,0.8)"),
        )
        st.plotly_chart(fig_tier, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── high-risk customers table ─────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🚨 High-Risk Customer Table</h2>
<p>Filter, sort, and download customers by risk level</p></div>""", unsafe_allow_html=True)

filter_col, _, dl_col1, dl_col2 = st.columns([1, 2, 1, 1])
with filter_col:
    risk_filter = st.selectbox(
        "Filter by Risk",
        options=["All", "High Risk", "Medium Risk", "Low Risk"],
        index=0,
    )

if risk_filter == "All":
    filtered = scored.copy()
else:
    filtered = scored[scored["Risk_Category"] == risk_filter].copy()

display_cols = ["Loyalty Number", "Loyalty_Card", "CLV", "Total_Revenue",
                "Total_Flights", "Churn_Pct", "Risk_Category"]
available = [c for c in display_cols if c in filtered.columns]
disp = filtered[available].copy()
disp = disp.rename(columns={
    "Loyalty_Card": "Loyalty Card",
    "Total_Revenue": "Revenue ($)",
    "Total_Flights": "Flights",
    "Churn_Pct": "Churn Probability (%)",
    "Risk_Category": "Risk",
})
if "CLV" in disp.columns:
    disp["CLV"] = disp["CLV"].map("${:,.0f}".format)
if "Revenue ($)" in disp.columns:
    disp["Revenue ($)"] = disp["Revenue ($)"].map("${:,.0f}".format)

disp = disp.sort_values("Churn Probability (%)", ascending=False).head(500)
st.dataframe(disp, use_container_width=True, hide_index=True)

# Downloads
with dl_col1:
    high_risk_df = scored[scored["Risk_Category"] == "High Risk"].copy()
    st.download_button(
        "⬇ High-Risk CSV",
        data=high_risk_df.to_csv(index=False).encode(),
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )

with dl_col2:
    report_df = scored[available].copy()
    st.download_button(
        "⬇ Full Prediction Report",
        data=report_df.to_csv(index=False).encode(),
        file_name="churn_prediction_report.csv",
        mime="text/csv",
    )

st.markdown("<br/>", unsafe_allow_html=True)

# ── risk vs clv scatter ────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🎯 Churn Risk × CLV Quadrant</h2>
<p>Target high CLV / high risk customers for maximum retention ROI</p></div>""",
            unsafe_allow_html=True)

if "CLV" in scored.columns and "Loyalty_Card" in scored.columns:
    sample_s = scored.dropna(subset=["CLV"]).sample(min(3000, len(scored)), random_state=7)
    fig_scatter = px.scatter(
        sample_s, x="Churn_Pct", y="CLV",
        color="Loyalty_Card",
        color_discrete_map=PALETTE,
        opacity=0.5,
        labels={"Churn_Pct": "Churn Probability (%)", "CLV": "Customer Lifetime Value ($)"},
        title="Risk vs CLV — Target top-right quadrant for retention",
    )
    fig_scatter.add_vline(x=70, line_dash="dash", line_color="#fc8181",
                          annotation_text="High Risk Threshold",
                          annotation_font_color="#fc8181")
    fig_scatter.add_hline(y=sample_s["CLV"].median(), line_dash="dash",
                          line_color="#68d391",
                          annotation_text="Median CLV",
                          annotation_font_color="#68d391")
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#a0aec0", height=460,
        legend=dict(bgcolor="rgba(13,21,38,0.8)"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── predictive insights ────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>💡 Predictive Insights</h2>
<p>Automated business intelligence from the churn model</p></div>""", unsafe_allow_html=True)

high_n   = (scored["Risk_Category"] == "High Risk").sum()
med_n    = (scored["Risk_Category"] == "Medium Risk").sum()
low_n    = (scored["Risk_Category"] == "Low Risk").sum()
total_n  = len(scored)
pct_high = high_n / total_n * 100

top_fi_idx  = np.argmax(model.feature_importances_)
top_feature = feature_names[top_fi_idx].replace("num__", "").replace("cat__", "").replace("_", " ").title()

avg_clv_high = scored[scored["Risk_Category"] == "High Risk"]["CLV"].mean() if "CLV" in scored.columns else 0
avg_clv_low  = scored[scored["Risk_Category"] == "Low Risk"]["CLV"].mean() if "CLV" in scored.columns else 0

insights = [
    ("🎯", f"The churn model achieves <strong>ROC-AUC = {saved_metrics['ROC_AUC']:.3f}</strong> with "
           f"<strong>{saved_metrics['Accuracy']*100:.1f}%</strong> accuracy — production-grade discrimination."),
    ("⚠️", f"<strong>{high_n:,}</strong> customers ({pct_high:.1f}%) are classified as <strong>High Risk</strong>. "
           f"An additional <strong>{med_n:,}</strong> are at Medium Risk and require monitoring."),
    ("🔑", f"The most predictive churn signal is <strong>{top_feature}</strong>. "
           f"Retention strategies should directly address this behavioural driver."),
    ("💰", f"High-risk customers have an average CLV of <strong>${avg_clv_high:,.0f}</strong> vs "
           f"<strong>${avg_clv_low:,.0f}</strong> for low-risk — making retention campaigns highly ROI-positive."),
    ("🚀", "Prioritise customers in the <em>High Risk + High CLV</em> quadrant for personalised "
           "offers — they represent the greatest recoverable lifetime revenue per retention dollar spent."),
    ("📋", f"With <strong>{saved_metrics['Recall']:.1%}</strong> recall, the model successfully identifies "
           f"the majority of actual churners — minimising missed at-risk customers."),
]
for icon, text in insights:
    st.markdown(f"""<div class='insight-card'>
    <div class='insight-icon'>{icon}</div>
    <div class='insight-text'>{text}</div></div>""", unsafe_allow_html=True)
