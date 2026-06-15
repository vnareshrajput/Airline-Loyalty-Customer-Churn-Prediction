# ✈ Airline Customer Loyalty Analytics Dashboard

> A Fortune-500-grade business intelligence platform built with **Streamlit**, **Plotly**, and **Scikit-Learn** — integrating an end-to-end machine learning churn prediction system with a 5-page interactive analytics dashboard.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Dashboard Pages](#dashboard-pages)
- [Churn Prediction Model](#churn-prediction-model)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Deployment](#deployment)
- [Key Results](#key-results)

---

## Project Overview

This project analyses a Canadian airline's customer loyalty programme covering **16,737 unique customers** across **389,065 transaction records** (2017–2018). It delivers:

- A **5-page interactive Streamlit dashboard** with executive KPIs, customer segmentation, loyalty analytics, revenue insights, and ML-powered churn prediction
- A **production-ready churn prediction pipeline** using RandomForest with SMOTE balancing, achieving **97.1% accuracy** and **ROC-AUC of 0.981**
- **Automated business insights** generated from real model outputs — risk badges, gauge charts, downloadable reports, and retention recommendations

---

## Dataset

| Property | Value |
|---|---|
| Source file | `master_airline_loyalty.csv` |
| Total records | 389,065 |
| Unique customers | 16,737 |
| Date range | January 2017 – December 2018 |
| Loyalty tiers | Aurora · Nova · Star |
| Provinces covered | 11 Canadian provinces |
| Overall churn rate | 12.4% |
| Average CLV | $7,987 |
| Features | 21 raw → 87 engineered |

**Supporting datasets:**

- `airline_segmented.csv` — Feature-engineered dataset with OHE columns (46 columns) used for model training and inference
- `airline_feature_engineered.csv` — Intermediate feature engineering output (44 columns)

---

## Project Structure

```
airline_dashboard/
│
├── app.py                          # Main entry point / Home page
│
├── pages/
│   ├── 1_Executive_Summary.py      # KPIs, revenue trends, tier overview
│   ├── 2_Customer_Analytics.py     # Demographics, segmentation, filters
│   ├── 3_Loyalty_Program.py        # Tier performance, points, retention
│   ├── 4_Revenue_Insights.py       # Waterfall, Pareto, Treemap
│   └── 5_Predictive_Analytics.py   # ML churn model + risk scoring
│
├── models/
│   ├── churn_model.pkl             # Trained RandomForest classifier
│   ├── preprocessing_pipeline.pkl  # ColumnTransformer (scaler + OHE)
│   ├── feature_columns.pkl         # Column metadata (num/cat/bool lists)
│   ├── feature_names.pkl           # 87 post-transform feature names
│   └── model_metrics.pkl           # Saved evaluation metrics
│
├── utils/
│   ├── data_loader.py              # Cached data loading & feature derivation
│   ├── charts.py                   # All Plotly chart factory functions
│   └── kpi_calculator.py           # KPI aggregation functions
│
├── assets/
│   └── style.css                   # Enterprise dark theme (Inter font)
│
├── master_airline_loyalty.csv      # Primary dataset
├── airline_segmented.csv           # Model training dataset
├── airline_feature_engineered.csv  # Feature engineering output
├── 05_Churn_Prediction.ipynb       # Model development notebook
├── requirements.txt
└── README.md
```

---

## Dashboard Pages

### 🏠 Home
Executive overview with 6 KPI cards, navigation guide, dataset snapshot, and auto-generated highlights.

### 1 · 📊 Executive Summary
- 6 top-line KPI cards (Revenue, CLV, Flights, Churn %, Active Members)
- Monthly revenue trend line chart
- Loyalty funnel, tier revenue bar, customer donut, revenue waterfall
- Top provinces by revenue, CLV distribution histogram
- Tier deep-dive table + 5 auto-generated executive insights

### 2 · 👥 Customer Analytics
- **6 dynamic sidebar filters**: Tier, Gender, Education, Province, CLV range, Salary range
- Demographics: gender donut, education bar, marital status pie
- Segmentation tabs: CLV, Salary, Engagement, Flights
- Geographic heatmap, behavioural scatter plots, correlation matrix
- Income group analysis, Education × Tier revenue sunburst

### 3 · 🏆 Loyalty Program
- Points KPIs: Accumulated (total), Redeemed, Redemption Rate, Balance
- Points accumulated vs redeemed by tier (grouped bar)
- Churn rate by tier + Tier × Marital Status heatmap + Enrollment year trend
- Engagement box plots, Education engagement bar
- Customer flow Sankey diagram (Education → Tier → Status)
- Monthly points accumulation timeline
- 5 data-driven loyalty recommendations

### 4 · 💰 Revenue Insights
- Revenue breakdown charts: waterfall, Pareto, treemap, decomposition
- Province and tier revenue analysis

### 5 · 🤖 Predictive Analytics *(ML-Powered)*
- **Model Performance cards**: ROC-AUC, Accuracy, Precision, Recall, F1
- **Top 15 Feature Importances** — horizontal bar chart
- **Individual Customer Predictor**: searchable dropdown, Plotly gauge, risk badge
- **Auto-generated retention insight** per customer
- **Risk segmentation charts**: category bar, probability histogram, risk-by-tier
- **Filterable high-risk table** (High / Medium / Low Risk, top 500 rows)
- **Churn Risk × CLV quadrant scatter** with annotation lines
- **2 download buttons**: High-Risk CSV, Full Prediction Report CSV
- **6 automated business insights** from model output

---

## Churn Prediction Model

### Pipeline (replicates notebook exactly)

```
airline_segmented.csv
        ↓
Drop: ['Cancelled', 'Loyalty Number']
Convert bool columns → int
        ↓
ColumnTransformer
  ├── Numeric (41 cols): SimpleImputer(median) → StandardScaler
  └── Categorical (4 cols): SimpleImputer(most_frequent) → OneHotEncoder
        ↓
train_test_split (80/20, stratified, random_state=42)
        ↓
SMOTE (random_state=42) — balances minority class (~12%)
        ↓
RandomForestClassifier(n_estimators=100, max_depth=30,
                       min_samples_leaf=3, random_state=42)
        ↓
Saved to models/ — loaded at startup, no retraining
```

### Performance Metrics

| Metric | Score |
|---|---|
| **ROC-AUC** | **0.981** |
| **Accuracy** | **97.1%** |
| **Precision** | 94.5% |
| **Recall** | 81.6% |
| **F1 Score** | 87.5% |

### Top Predictive Features

| Rank | Feature | Importance |
|---|---|---|
| 1 | Enrollment Year | 0.116 |
| 2 | Membership Age (Months) | 0.079 |
| 3 | Total Flights | 0.073 |
| 4 | Tenure | 0.070 |
| 5 | CLV | 0.051 |

### Risk Classification

| Range | Category |
|---|---|
| 0 – 40% | 🟢 Low Risk |
| 40 – 70% | 🟠 Medium Risk |
| 70 – 100% | 🔴 High Risk |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit ≥ 1.35 |
| Visualisation | Plotly ≥ 5.18 |
| Data | Pandas ≥ 2.0, NumPy ≥ 1.24 |
| ML | Scikit-learn ≥ 1.4, imbalanced-learn ≥ 0.12 |
| Model persistence | Joblib ≥ 1.3 |
| Styling | Custom CSS — Inter font, dark navy enterprise theme |
| Language | Python 3.10+ |

---

## Quick Start

```bash
# 1. Clone / unzip the project
cd airline_dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

The `models/` folder is pre-populated — no retraining required on startup.

---

## Deployment

### Streamlit Community Cloud

1. Push the project folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo and set **Main file path** to `app.py`
4. Click **Deploy**

> **Note:** The `models/churn_model.pkl` file (~45 MB) must be committed to the repo. Ensure your Git LFS or repo size limits allow this.

### requirements.txt

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.4.0
imbalanced-learn>=0.12.0
joblib>=1.3.0
```

---

## Key Results

| Metric | Value |
|---|---|
| Customers analysed | 16,737 |
| Overall churn rate | 12.4% |
| High-risk customers | ~2,419 (14.5%) |
| Aurora tier avg CLV | $10,673 |
| Star tier avg CLV | $6,733 |
| Model ROC-AUC | 0.981 |
| Model accuracy | 97.1% |
| Top churn predictor | Enrollment Year |

---

*Built by Naresh V · IIT Roorkee · Data Science & Analytics*
