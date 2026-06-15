"""
pages/5_Predictive_Analytics.py
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

st.set_page_config(page_title="Predictive Analytics", page_icon="🤖", layout="wide")
css_path = os.path.join(ROOT, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from utils.data_loader import load_data, get_customer_df
from utils.charts import roc_curve_chart, feature_importance_bar, QUAL_COLORS, PALETTE

@st.cache_resource
def _load():
    p = os.path.join(ROOT, "master_airline_loyalty.csv")
    df = load_data(p); cdf = get_customer_df(df)
    return df, cdf

df, cdf = _load()

st.markdown("""
<div class='page-title'>
  <h1>🤖 Predictive Analytics</h1>
  <p>Machine learning models for churn prediction and CLV forecasting</p>
</div>""", unsafe_allow_html=True)

# ── feature engineering ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Engineering features…")
def prepare_features(cdf: pd.DataFrame):
    from sklearn.preprocessing import LabelEncoder
    ml = cdf.copy()

    cat_cols = ["Gender","Education","Loyalty_Card","Marital_Status"]
    le = LabelEncoder()
    for c in cat_cols:
        ml[c+"_enc"] = le.fit_transform(ml[c].astype(str))

    feat_cols = [
        "Total_Flights","Total_Distance","Total_Points","Points_Redeemed",
        "Total_Revenue","CLV","Salary","Engagement","Efficiency",
        "Months_Active","Redemption_Rate","Revenue_Per_Customer",
        "Gender_enc","Education_enc","Loyalty_Card_enc","Marital_Status_enc",
    ]
    ml = ml.dropna(subset=feat_cols+["Cancelled"])
    X = ml[feat_cols].values
    y_churn = ml["Cancelled"].values
    y_clv   = ml["CLV"].values
    return X, y_churn, y_clv, feat_cols, ml


@st.cache_resource(show_spinner="Training models…")
def train_churn_model(cdf_hash):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (roc_auc_score, accuracy_score,
                                  classification_report, roc_curve)
    from sklearn.preprocessing import StandardScaler

    X, y_churn, _, feat_cols, ml = prepare_features(cdf)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_churn, test_size=0.2,
                                                random_state=42, stratify=y_churn)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                        learning_rate=0.08, random_state=42)
    model.fit(X_tr, y_tr)

    y_pred   = model.predict(X_te)
    y_prob   = model.predict_proba(X_te)[:,1]
    auc      = roc_auc_score(y_te, y_prob)
    acc      = accuracy_score(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    report   = classification_report(y_te, y_pred, output_dict=True)
    return model, scaler, feat_cols, auc, acc, fpr, tpr, report


@st.cache_resource(show_spinner="Training CLV model…")
def train_clv_model(cdf_hash):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    X, _, y_clv, feat_cols, ml = prepare_features(cdf)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_clv, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                       learning_rate=0.08, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    r2     = r2_score(y_te, y_pred)
    mae    = mean_absolute_error(y_te, y_pred)
    return model, scaler, feat_cols, r2, mae, y_te, y_pred


# ── train section ─────────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>⚙️ Model Training Controls</h2></div>""",
            unsafe_allow_html=True)

train_col, info_col = st.columns([1, 2])
with train_col:
    train_btn = st.button("🚀 Train / Refresh Models", type="primary",
                          help="Train Gradient Boosting models for churn & CLV prediction")
with info_col:
    st.markdown("""
    <div style='color:#a0aec0; font-size:13px; padding:8px 0;'>
    Models train on <strong style='color:#63b3ed'>80%</strong> of customer data and evaluate
    on <strong style='color:#63b3ed'>20%</strong> holdout. Using
    <strong style='color:#63b3ed'>Gradient Boosting</strong> for both tasks.
    </div>""", unsafe_allow_html=True)

cdf_hash = hash(len(cdf))

with st.spinner("Training…"):
    churn_model, churn_scaler, feat_cols, auc, acc, fpr, tpr, report = train_churn_model(cdf_hash)
    clv_model,   clv_scaler,   feat_cols, r2,  mae, y_te, y_pred      = train_clv_model(cdf_hash)

# ── model KPI cards ───────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>📊 Model Performance</h2></div>""",
            unsafe_allow_html=True)

cols = st.columns(6)
kpi_list = [
    ("🎯","Churn AUC",   f"{auc:.3f}",       "positive" if auc>0.8 else "neutral"),
    ("✅","Churn Acc.",  f"{acc*100:.1f}%",   "positive" if acc>0.8 else "neutral"),
    ("📈","CLV R²",      f"{r2:.3f}",         "positive" if r2>0.8 else "neutral"),
    ("💵","CLV MAE",     f"${mae:,.0f}",      "neutral"),
    ("⚡","Precision",   f"{report['1']['precision']:.3f}","neutral"),
    ("🔄","Recall",      f"{report['1']['recall']:.3f}",  "neutral"),
]
for col,(icon,lbl,val,cls) in zip(cols, kpi_list):
    with col:
        st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-icon'>{icon}</div>
        <div class='kpi-label'>{lbl}</div>
        <div class='kpi-value'>{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── ROC + Feature Importance ──────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🔵 Churn Model Diagnostics</h2></div>""",
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(roc_curve_chart(fpr, tpr, auc), use_container_width=True)
with c2:
    fi  = churn_model.feature_importances_
    st.plotly_chart(feature_importance_bar(feat_cols, fi), use_container_width=True)

# ── CLV model: actual vs predicted ───────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>💰 CLV Prediction — Actual vs Predicted</h2></div>""",
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    sample_idx = np.random.choice(len(y_te), size=min(2000, len(y_te)), replace=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_te[sample_idx], y=y_pred[sample_idx],
        mode="markers", opacity=0.4,
        marker=dict(color="#63b3ed", size=4),
        name="Predicted",
    ))
    max_val = max(y_te.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode="lines",
        line=dict(color="#fc8181", dash="dash"), name="Perfect Fit",
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=420,
                      title=f"CLV: Actual vs Predicted (R²={r2:.3f})",
                      xaxis_title="Actual CLV ($)", yaxis_title="Predicted CLV ($)",
                      margin=dict(l=40,r=20,t=50,b=40))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    clv_fi = clv_model.feature_importances_
    st.plotly_chart(feature_importance_bar(feat_cols, clv_fi), use_container_width=True)

# ── full-dataset predictions ──────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>⚠️ Risk Segmentation</h2></div>""",
            unsafe_allow_html=True)

@st.cache_data(show_spinner="Scoring all customers…")
def score_all(_churn_model, _churn_scaler, _clv_model, _clv_scaler, feat_cols):
    X, _, _, _, ml = prepare_features(cdf)
    X_scaled = _churn_scaler.transform(X)
    churn_prob = _churn_model.predict_proba(X_scaled)[:,1]
    clv_pred   = _clv_model.predict(_clv_scaler.transform(X))

    result = ml[["Loyalty Number","Loyalty_Card","Province","Gender","CLV","Total_Flights"]].copy()
    result["Churn_Probability"] = np.round(churn_prob, 4)
    result["Predicted_CLV"]     = np.round(clv_pred, 2)
    result["Risk_Segment"] = pd.cut(
        churn_prob,
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Very Low", "Low", "Medium", "High", "Critical"],
    )
    return result

scored = score_all(churn_model, churn_scaler, clv_model, clv_scaler, feat_cols)

# risk distribution
c1, c2, c3 = st.columns(3)
with c1:
    risk_counts = scored["Risk_Segment"].value_counts().reset_index()
    risk_counts.columns = ["Segment","Count"]
    order  = ["Very Low","Low","Medium","High","Critical"]
    colors = ["#68d391","#4fd1c5","#f6ad55","#fc8181","#9f7aea"]
    risk_counts = risk_counts.set_index("Segment").reindex(order).fillna(0).reset_index()
    fig = go.Figure(go.Bar(
        x=risk_counts["Segment"], y=risk_counts["Count"],
        marker_color=colors,
        text=risk_counts["Count"].astype(int),
        textposition="outside",
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=380,
                      title="Customer Count by Risk Segment",
                      margin=dict(l=20,r=20,t=50,b=40))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Churn probability histogram
    fig = px.histogram(scored, x="Churn_Probability", nbins=50,
                       color_discrete_sequence=[QUAL_COLORS[0]],
                       title="Churn Probability Distribution")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=380)
    st.plotly_chart(fig, use_container_width=True)

with c3:
    # risk by tier
    risk_tier = scored.groupby(["Loyalty_Card","Risk_Segment"])["Loyalty Number"].count().reset_index()
    risk_tier.columns = ["Tier","Risk","Count"]
    fig = px.bar(risk_tier, x="Tier", y="Count", color="Risk",
                 color_discrete_sequence=colors,
                 title="Risk Segment by Loyalty Tier",
                 category_orders={"Risk":order})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=380,
                      legend=dict(bgcolor="rgba(13,21,38,0.8)"))
    st.plotly_chart(fig, use_container_width=True)

# ── risk vs predicted CLV scatter ────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🎯 Churn Risk × Predicted CLV</h2></div>""",
            unsafe_allow_html=True)

sample = scored.sample(min(3000, len(scored)), random_state=7)
fig = px.scatter(
    sample, x="Churn_Probability", y="Predicted_CLV",
    color="Loyalty_Card", color_discrete_map=PALETTE,
    opacity=0.5, size_max=8,
    labels={"Churn_Probability":"Churn Risk","Predicted_CLV":"Predicted CLV ($)"},
    title="Risk vs CLV Quadrant — Target high CLV / high risk customers",
)
fig.add_vline(x=0.5, line_dash="dash", line_color="#fc8181",
              annotation_text="Risk Threshold", annotation_font_color="#fc8181")
fig.add_hline(y=sample["Predicted_CLV"].median(), line_dash="dash",
              line_color="#68d391", annotation_text="Median CLV",
              annotation_font_color="#68d391")
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font_color="#a0aec0", height=460,
                  legend=dict(bgcolor="rgba(13,21,38,0.8)"))
st.plotly_chart(fig, use_container_width=True)

# ── top at-risk customers table ────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>🚨 High-Risk, High-Value Customers</h2>
<p>Customers with churn probability &gt; 0.6 AND predicted CLV &gt; median — priority retention targets</p></div>""",
            unsafe_allow_html=True)

median_clv = scored["Predicted_CLV"].median()
priority   = (
    scored[(scored["Churn_Probability"] > 0.6) & (scored["Predicted_CLV"] > median_clv)]
    .sort_values("Predicted_CLV", ascending=False)
    .head(50)
)

disp_cols = ["Loyalty Number","Loyalty_Card","Province","Gender",
             "CLV","Total_Flights","Churn_Probability","Predicted_CLV","Risk_Segment"]
priority_disp = priority[disp_cols].copy()
priority_disp["CLV"]               = priority_disp["CLV"].map("${:,.0f}".format)
priority_disp["Predicted_CLV"]     = priority_disp["Predicted_CLV"].map("${:,.0f}".format)
priority_disp["Churn_Probability"] = priority_disp["Churn_Probability"].map("{:.1%}".format)
st.dataframe(priority_disp, use_container_width=True, hide_index=True)

# ── model insights ────────────────────────────────────────────────────────────
st.markdown("""<div class='section-header'><h2>💡 Predictive Insights</h2></div>""",
            unsafe_allow_html=True)

critical_n   = (scored["Risk_Segment"] == "Critical").sum()
high_n       = (scored["Risk_Segment"] == "High").sum()
top_feature  = feat_cols[np.argmax(churn_model.feature_importances_)]
pct_high_risk = (critical_n + high_n) / len(scored) * 100

insights = [
    ("🎯", f"The churn model achieves <strong>AUC={auc:.3f}</strong> — excellent discrimination "
           f"between churners and loyal customers."),
    ("⚠️", f"<strong>{critical_n + high_n:,}</strong> customers (<strong>{pct_high_risk:.1f}%</strong>) "
           "are in High or Critical risk segments and should be prioritised for retention."),
    ("🔑", f"The most predictive churn feature is <strong>{top_feature.replace('_',' ')}</strong>. "
           "Focus retention messaging on this behavioural driver."),
    ("💵", f"CLV model R² = <strong>{r2:.3f}</strong>, MAE = <strong>${mae:,.0f}</strong>. "
           "Accurate CLV forecasting enables targeted high-ROI retention spend."),
    ("🚀", "Prioritise <em>High Risk + High CLV</em> quadrant customers for personalised "
           "retention offers — they represent the greatest recoverable revenue."),
]
for icon, text in insights:
    st.markdown(f"""<div class='insight-card'>
    <div class='insight-icon'>{icon}</div>
    <div class='insight-text'>{text}</div></div>""", unsafe_allow_html=True)

# ── downloads ─────────────────────────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇ Download All Scored Customers",
        scored.to_csv(index=False).encode(),
        "all_scored_customers.csv","text/csv",
    )
with c2:
    st.download_button(
        "⬇ Download Priority Retention List",
        priority[disp_cols].to_csv(index=False).encode(),
        "priority_retention.csv","text/csv",
    )
