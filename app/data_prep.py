"""
Reusable cleaning and feature engineering logic for Project 1, Supply Chain.

This mirrors the steps worked out in notebook 01_data_prep.ipynb, packaged as
functions so the Streamlit app can run the same logic on a freshly uploaded
CSV, not just the original dataset.
"""

import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "date", "sku_id", "category", "units_sold",
    "units_received", "closing_stock", "lead_time_days",
]

FEATURE_COLUMNS = [
    "sold_lag_1", "sold_lag_7", "sold_roll_mean_7",
    "sold_roll_mean_14", "demand_trend", "day_of_week",
]

NEW_SKU_HISTORY_THRESHOLD = 30


def validate_columns(df):
    """Check that an uploaded CSV has the columns this pipeline needs."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing required columns: {missing}")


def clean_data(df):
    """Apply the same fixes found during exploration in notebook 1."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Standardize category casing
    df["category"] = df["category"].str.strip().str.title()

    # Each SKU should have one fixed lead time, use the most common value
    lead_time_fix = df.groupby("sku_id")["lead_time_days"].agg(lambda x: x.mode()[0])
    df["lead_time_days"] = df["sku_id"].map(lead_time_fix)

    # Fill missing sales and stock values per SKU using the last known value
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    df["units_sold"] = df.groupby("sku_id")["units_sold"].transform(lambda x: x.ffill().bfill())
    df["closing_stock"] = df.groupby("sku_id")["closing_stock"].transform(lambda x: x.ffill().bfill())

    return df


def split_new_and_established(df):
    """Separate SKUs with enough history for forecasting from newly launched ones."""
    sku_counts = df.groupby("sku_id")["date"].count()
    new_skus = sku_counts[sku_counts < NEW_SKU_HISTORY_THRESHOLD].index.tolist()
    df = df.copy()
    df["is_new_sku"] = df["sku_id"].isin(new_skus)
    return df, new_skus


def add_features(df):
    """Build the lag, rolling average, trend, and days-of-cover features."""
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    df["sold_lag_1"] = df.groupby("sku_id")["units_sold"].shift(1)
    df["sold_lag_7"] = df.groupby("sku_id")["units_sold"].shift(7)

    df["sold_roll_mean_7"] = df.groupby("sku_id")["units_sold"].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    df["sold_roll_mean_14"] = df.groupby("sku_id")["units_sold"].transform(
        lambda x: x.shift(1).rolling(14).mean()
    )

    df["sold_roll_mean_7_prev"] = df.groupby("sku_id")["sold_roll_mean_7"].shift(7)
    df["demand_trend"] = df["sold_roll_mean_7"] - df["sold_roll_mean_7_prev"]

    df["day_of_week"] = df["date"].dt.dayofweek

    df["days_of_cover"] = df["closing_stock"] / df["sold_roll_mean_7"].replace(0, np.nan)

    return df


def build_established_flags(df, model):
    """Score established SKUs with the trained model and apply the risk rule."""
    latest = df.sort_values("date").groupby("sku_id").tail(1).copy()
    latest = latest.dropna(subset=FEATURE_COLUMNS)

    if latest.empty:
        return latest

    latest["forecasted_demand"] = model.predict(latest[FEATURE_COLUMNS])
    latest["days_of_cover"] = latest["closing_stock"] / latest["forecasted_demand"].replace(0, np.nan)
    latest["risk_flag"] = latest.apply(_flag_risk, axis=1)
    latest["confidence"] = "Model-based"
    return latest


def _flag_risk(row):
    if row["days_of_cover"] < row["lead_time_days"]:
        return "stockout risk"
    if row["days_of_cover"] > 3 * row["lead_time_days"] and row["demand_trend"] <= 0:
        return "overstock risk"
    return "ok"


def build_new_sku_flags(new_sku_df):
    """Apply the simple average-based rule for SKUs with too little history to model."""
    summary = new_sku_df.groupby("sku_id").agg(
        avg_daily_sales=("units_sold", "mean"),
        latest_stock=("closing_stock", "last"),
        lead_time_days=("lead_time_days", "last"),
        category=("category", "last"),
        days_of_history=("date", "count"),
    ).reset_index()

    summary["days_of_cover"] = summary["latest_stock"] / summary["avg_daily_sales"].replace(0, np.nan)
    summary["risk_flag"] = np.where(
        summary["days_of_cover"] < summary["lead_time_days"],
        "stockout risk",
        "ok",
    )
    summary["confidence"] = "Early estimate, limited history"
    return summary


def run_full_pipeline(raw_df, model):
    """
    Take a raw CSV in the expected format, all the way through cleaning,
    feature engineering, and risk flagging. Used both for the bundled
    dataset at startup and for any freshly uploaded CSV.
    """
    validate_columns(raw_df)
    cleaned = clean_data(raw_df)
    split, new_skus = split_new_and_established(cleaned)

    established_part = split[~split["is_new_sku"]]
    new_part = split[split["is_new_sku"]]

    established_features = add_features(established_part) if not established_part.empty else established_part
    established_flags = (
        build_established_flags(established_features, model)
        if not established_features.empty else established_features
    )
    new_flags = build_new_sku_flags(new_part) if not new_part.empty else new_part

    return established_flags, new_flags
