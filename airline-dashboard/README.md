# ✈ Airline Customer Loyalty Analytics Dashboard

A Fortune-500-grade, enterprise-quality Streamlit dashboard for airline loyalty program intelligence.

---

## 🚀 Quick Start

```bash
# 1. Clone / unzip the project
cd airline_dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your dataset
cp /path/to/master_airline_loyalty.csv .

# 4. Run
streamlit run app.py
```

---

## 📁 Project Structure

```
airline_dashboard/
├── app.py                       ← Home / entry point
├── requirements.txt
├── README.md
├── master_airline_loyalty.csv   ← Dataset (place here)
│
├── pages/
│   ├── 1_Executive_Summary.py
│   ├── 2_Customer_Analytics.py
│   ├── 3_Loyalty_Program.py
│   ├── 4_Revenue_Insights.py
│   └── 5_Predictive_Analytics.py
│
├── assets/
│   └── style.css
│
└── utils/
    ├── __init__.py
    ├── data_loader.py
    ├── kpi_calculator.py
    └── charts.py
```

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Overview KPIs, navigation guide, tier composition |
| 📊 Executive Summary | Revenue trends, tier analysis, province rankings |
| 👥 Customer Analytics | Demographics, segmentation, correlation analysis |
| 🏆 Loyalty Program | Points, redemption, churn, retention matrix, Sankey |
| 💰 Revenue Insights | Waterfall, Pareto, Treemap, Sunburst, heatmaps |
| 🤖 Predictive Analytics | Churn + CLV models, ROC curve, risk scoring |

---

## ✨ Features

- **Dark enterprise theme** with glassmorphism KPI cards
- **Interactive filters** on every page (tier, gender, province, CLV range)
- **Auto-generated insights** after every visualization
- **ML models**: Gradient Boosting for churn + CLV with full diagnostics
- **CSV upload** to swap datasets at runtime
- **Downloadable** predictions, filtered data, and reports
- **@st.cache_data / @st.cache_resource** for performance

---

## 🛠 Tech Stack

- **Streamlit** — UI framework
- **Plotly** — All visualizations (20+ chart types)
- **Pandas / NumPy** — Data processing
- **Scikit-Learn** — Machine learning models

---

## 📷 Screenshots

> Add screenshots here after first run.

---

## 📄 Dataset Columns

| Column | Description |
|--------|-------------|
| Loyalty Number | Unique customer ID |
| Year / Month | Transaction period |
| Total Flights | Flights in period |
| Distance | Miles flown |
| Points Accumulated / Redeemed | Loyalty points |
| Dollar Cost Points Redeemed | $ value of redeemed points |
| Country / Province / City | Geography |
| Gender / Education / Salary | Demographics |
| Loyalty Card | Tier (Aurora / Nova / Star) |
| CLV | Customer Lifetime Value |
| Enrollment Year / Month | When customer joined |
| Cancelled | 1 = cancelled membership |

---

## 📝 License

For portfolio and demonstration purposes.
