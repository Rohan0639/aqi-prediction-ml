"""
preprocessing.py
================
Data cleaning pipeline:
  - Parse dates
  - Remove duplicates
  - Handle missing values
  - Clip extreme outliers (IQR method)
  - Sort by Station + Date for lag-feature correctness
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Numeric columns that should be cleaned / clipped
NUMERIC_COLS = [
    "AQI", "PM2.5", "PM10", "NO2", "SO2", "O3", "CO",
    "Temperature", "Humidity", "Wind_Speed", "Rainfall",
]


def _clip_outliers(df: pd.DataFrame, cols: list, factor: float = 3.0) -> pd.DataFrame:
    """
    Clip values beyond [Q1 - factor*IQR, Q3 + factor*IQR] for each column.
    Uses a conservative factor=3 to not destroy real pollution spikes.
    """
    for col in cols:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        before = df[col].between(lower, upper).sum()
        df[col] = df[col].clip(lower, upper)
        logger.debug(f"  {col}: clipped to [{lower:.2f}, {upper:.2f}] ({before} in-range)")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline.

    Parameters
    ----------
    df : pd.DataFrame  – raw DataFrame from data_loader

    Returns
    -------
    pd.DataFrame – cleaned DataFrame, sorted by Station + Date
    """
    logger.info("── Starting data cleaning ──────────────────────")
    original_shape = df.shape

    # 1. Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
        logger.info("Converted 'Date' column to datetime.")

    # 2. Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["Date", "Station"])
    removed = before - len(df)
    if removed:
        logger.info(f"Removed {removed} duplicate rows (Date + Station).")

    # 3. Enforce numeric types
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Handle missing values
    missing_before = df[NUMERIC_COLS].isnull().sum().sum()
    if missing_before > 0:
        logger.info(f"Missing values before imputation: {missing_before}")
        # Forward-fill within each station group, then backfill any residual
        df = df.sort_values(["Station", "Date"])
        df[NUMERIC_COLS] = (
            df.groupby("Station")[NUMERIC_COLS]
            .transform(lambda g: g.ffill().bfill())
        )
        # Any remaining NaNs (e.g. entire station missing) → global median
        for col in NUMERIC_COLS:
            if df[col].isnull().any():
                med = df[col].median()
                df[col] = df[col].fillna(med)
                logger.info(f"  '{col}': filled remaining NaN with global median {med:.2f}")

    # 5. Clip outliers
    logger.info("Clipping outliers (IQR × 3)…")
    df = _clip_outliers(df, NUMERIC_COLS)

    # 6. Ensure non-negative values where physically impossible
    for col in ["AQI", "PM2.5", "PM10", "NO2", "SO2", "O3", "CO",
                "Humidity", "Wind_Speed", "Rainfall"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # 7. Sort for lag-feature correctness
    df = df.sort_values(["Station", "Date"]).reset_index(drop=True)

    logger.info(f"Shape: {original_shape} → {df.shape}  "
                f"(removed {original_shape[0] - df.shape[0]} rows)")
    logger.info("── Data cleaning complete ───────────────────────")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    from src.data_loader import load_config, load_raw_data
    cfg = load_config()
    raw = load_raw_data(cfg)
    clean = clean_data(raw)
    print(clean.describe().round(2))
