"""
kpi_calculator.py — Dashboard KPI helpers
"""
import pandas as pd
import numpy as np


def executive_kpis(df: pd.DataFrame, cdf: pd.DataFrame) -> dict:
    active_customers   = cdf[cdf["Cancelled"] == 0]
    cancelled_customers= cdf[cdf["Cancelled"] == 1]

    total_customers  = len(cdf)
    total_revenue    = cdf["Total_Revenue"].sum()
    avg_clv          = cdf["CLV"].mean()
    avg_flights      = cdf["Total_Flights"].mean()
    active_count     = len(active_customers)
    churn_pct        = len(cancelled_customers) / total_customers * 100 if total_customers else 0

    total_points_acc = cdf["Total_Points"].sum()
    total_points_red = cdf["Points_Redeemed"].sum()
    redemption_rate  = total_points_red / total_points_acc * 100 if total_points_acc else 0

    avg_engagement   = cdf["Engagement"].mean()

    tier_counts = cdf["Loyalty_Card"].value_counts()

    return {
        "total_customers":  total_customers,
        "total_revenue":    total_revenue,
        "avg_clv":          avg_clv,
        "avg_flights":      avg_flights,
        "active_members":   active_count,
        "churn_pct":        churn_pct,
        "redemption_rate":  redemption_rate,
        "avg_engagement":   avg_engagement,
        "tier_counts":      tier_counts,
        "total_points_acc": total_points_acc,
        "total_points_red": total_points_red,
    }


def revenue_by_tier(cdf: pd.DataFrame) -> pd.DataFrame:
    return (
        cdf.groupby("Loyalty_Card")
           .agg(Revenue=("Total_Revenue","sum"), Customers=("Loyalty Number","count"))
           .reset_index()
           .rename(columns={"Loyalty_Card":"Tier"})
    )


def revenue_by_province(cdf: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    out = (
        cdf.groupby("Province")
           .agg(Revenue=("Total_Revenue","sum"), Customers=("Loyalty Number","count"))
           .reset_index()
           .sort_values("Revenue", ascending=False)
           .head(top_n)
    )
    return out


def revenue_monthly(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Date")["Revenue"]
          .sum()
          .reset_index()
          .sort_values("Date")
    )


def tier_performance(cdf: pd.DataFrame) -> pd.DataFrame:
    return (
        cdf.groupby("Loyalty_Card")
           .agg(
               Customers       = ("Loyalty Number",  "count"),
               Avg_CLV         = ("CLV",              "mean"),
               Avg_Flights     = ("Total_Flights",    "mean"),
               Total_Revenue   = ("Total_Revenue",    "sum"),
               Avg_Engagement  = ("Engagement",       "mean"),
               Churn_Rate      = ("Cancelled",        "mean"),
           )
           .reset_index()
           .rename(columns={"Loyalty_Card":"Tier"})
    )


def engagement_by_education(cdf: pd.DataFrame) -> pd.DataFrame:
    return (
        cdf.groupby("Education")
           .agg(
               Avg_Engagement = ("Engagement",       "mean"),
               Avg_CLV        = ("CLV",              "mean"),
               Avg_Flights    = ("Total_Flights",    "mean"),
               Count          = ("Loyalty Number",   "count"),
           )
           .reset_index()
    )


def churn_by_tier(cdf: pd.DataFrame) -> pd.DataFrame:
    return (
        cdf.groupby("Loyalty_Card")["Cancelled"]
           .agg(["sum","count","mean"])
           .reset_index()
           .rename(columns={"Loyalty_Card":"Tier","sum":"Churned","count":"Total","mean":"Churn_Rate"})
    )
