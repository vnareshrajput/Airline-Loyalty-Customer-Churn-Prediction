"""
data_loader.py — Cached data loading & preprocessing
"""
import pandas as pd
import numpy as np
import streamlit as st
import logging, os

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bin_age(s: pd.Series, labels=None) -> pd.Series:
    bins   = [0, 25, 35, 45, 55, 65, 120]
    labels = labels or ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    return pd.cut(s, bins=bins, labels=labels, right=False)


def _salary_group(s: pd.Series) -> pd.Series:
    bins   = [0, 40_000, 70_000, 100_000, 150_000, 1_000_000]
    labels = ["< $40K", "$40K–$70K", "$70K–$100K", "$100K–$150K", "$150K+"]
    return pd.cut(s, bins=bins, labels=labels, right=False)


def _loyalty_score(df: pd.DataFrame) -> pd.Series:
    cols = ["Total Flights", "Distance", "Points Accumulated", "CLV"]
    sub  = df[cols].copy()
    for c in cols:
        rng = sub[c].max() - sub[c].min()
        sub[c] = (sub[c] - sub[c].min()) / (rng if rng else 1)
    return (sub.mean(axis=1) * 100).round(2)


# --------------------------------------------------------------------------- #
# Main loader
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading dataset…", ttl=3600)
def load_data(path: str | None = None) -> pd.DataFrame:
    default = os.path.join(os.path.dirname(__file__), "..", "master_airline_loyalty.csv")
    filepath = path or default

    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        logger.error("Could not read CSV: %s", exc)
        raise

    # ---- basic cleaning --------------------------------------------------- #
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["CLV", "Loyalty Number"], inplace=True)

    # numeric coerce
    for col in ["Salary", "CLV", "Points Accumulated"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Salary", "CLV"], inplace=True)

    # ---- date field -------------------------------------------------------- #
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
    )

    # ---- derived metrics --------------------------------------------------- #
    # Revenue proxy (CLV / 12  * months active; Points cost)
    df["Revenue"]           = df["CLV"] / 12 + df["Dollar Cost Points Redeemed"]
    df["Points Balance"]    = df["Points Accumulated"] - df["Points Redeemed"]
    df["Redemption Rate"]   = np.where(
        df["Points Accumulated"] > 0,
        df["Points Redeemed"] / df["Points Accumulated"],
        0
    ).clip(0, 1)
    df["Flight Frequency Index"] = np.log1p(df["Total Flights"])
    df["Engagement Score"]       = (
        0.4 * df["Flight Frequency Index"]
        + 0.3 * (df["Redemption Rate"] * 10)
        + 0.3 * (df["Points Accumulated"] / (df["Points Accumulated"].max() or 1) * 10)
    ).round(2)
    df["Loyalty Efficiency Score"] = (
        df["Points Accumulated"] / (df["Distance"].replace(0, np.nan).fillna(1))
    ).round(4)

    # categorical derivations
    df["Enrollment Duration"] = (df["Year"] - df["Enrollment Year"]) * 12 + (df["Month"] - df["Enrollment Month"])
    df["Enrollment Duration"] = df["Enrollment Duration"].clip(0)

    df["Salary Group"] = _salary_group(df["Salary"])

    # status flag
    df["Status"] = np.where(df["Cancelled"] == 1, "Cancelled", "Active")

    return df


# --------------------------------------------------------------------------- #
# Customer-level aggregation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Aggregating customer data…", ttl=3600)
def get_customer_df(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("Loyalty Number").agg(
        Total_Flights   = ("Total Flights",    "sum"),
        Total_Distance  = ("Distance",         "sum"),
        Total_Points    = ("Points Accumulated","sum"),
        Points_Redeemed = ("Points Redeemed",  "sum"),
        Total_Revenue   = ("Revenue",          "sum"),
        CLV             = ("CLV",              "mean"),
        Salary          = ("Salary",           "mean"),
        Engagement      = ("Engagement Score", "mean"),
        Efficiency      = ("Loyalty Efficiency Score", "mean"),
        Gender          = ("Gender",           "first"),
        Education       = ("Education",        "first"),
        Province        = ("Province",         "first"),
        City            = ("City",             "first"),
        Loyalty_Card    = ("Loyalty Card",     "first"),
        Marital_Status  = ("Marital Status",   "first"),
        Salary_Group    = ("Salary Group",     "first"),
        Cancelled       = ("Cancelled",        "max"),
        Enrollment_Yr   = ("Enrollment Year",  "first"),
        Months_Active   = ("Enrollment Duration","max"),
    ).reset_index()

    grp["Revenue_Per_Customer"] = grp["Total_Revenue"] / grp["Total_Flights"].replace(0, 1)
    grp["Redemption_Rate"]      = (grp["Points_Redeemed"] / grp["Total_Points"].replace(0, 1)).clip(0, 1)
    grp["Churn_Risk_Score"]     = (
        (1 - grp["Redemption_Rate"]) * 0.4
        + (1 - (grp["Total_Flights"] / grp["Total_Flights"].max())) * 0.35
        + (grp["Cancelled"]) * 0.25
    ).clip(0, 1).round(4)
    grp["CLV_Segment"] = pd.qcut(grp["CLV"], q=4, labels=["Bronze", "Silver", "Gold", "Platinum"])
    return grp
