# ✈ Airline Customer Loyalty Analytics Dashboard

A Fortune-500-grade analytics platform built with Streamlit, Plotly, and Scikit-Learn.

## Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Executive KPIs & navigation overview |
| 1 · Executive Summary | Revenue trends, tier breakdown, top-line metrics |
| 2 · Customer Analytics | Demographics, segmentation & behavioural filters |
| 3 · Loyalty Program | Tier performance, points & retention analysis |
| 4 · Revenue Insights | Waterfall, Pareto, Treemap & decomposition |
| 5 · Predictive Analytics | **Churn prediction with trained Random Forest model** |

## Churn Model

- **Algorithm**: RandomForestClassifier (trained with SMOTE balancing)
- **Pipeline**: Median imputation → StandardScaler → OneHotEncoder → RF
- **ROC-AUC**: 0.947 | **Accuracy**: 90.8% | **Recall**: 78.7%
- Pre-trained artifacts stored in `models/` — no retraining on startup

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Fully compatible with **Streamlit Community Cloud**. Push to GitHub and connect the repo.

## Project Structure

```
airline_dashboard/
├── app.py                      # Main entry point
├── pages/
│   ├── 1_Executive_Summary.py
│   ├── 2_Customer_Analytics.py
│   ├── 3_Loyalty_Program.py
│   ├── 4_Revenue_Insights.py
│   └── 5_Predictive_Analytics.py   ← Updated with real churn model
├── models/
│   ├── churn_model.pkl             ← Trained RandomForest
│   ├── preprocessing_pipeline.pkl  ← ColumnTransformer (scaler + encoder)
│   ├── feature_columns.pkl         ← Column metadata
│   ├── feature_names.pkl           ← Post-transform feature names
│   └── model_metrics.pkl           ← Saved evaluation metrics
├── utils/
│   ├── data_loader.py
│   ├── charts.py
│   └── kpi_calculator.py
├── assets/style.css
├── master_airline_loyalty.csv
├── airline_segmented.csv
└── requirements.txt
```
