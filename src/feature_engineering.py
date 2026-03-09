"""
feature_engineering.py
=======================
Creates predictive features for next-day AQI:
  - Lag features   (AQI at t-1, t-2, t-3, t-7)
  - Rolling means  (3-day, 7-day, 14-day)
  - Seasonal flags (month, season, is_weekend)
  - Pollutant interaction features
  - Target column  (Next_day_AQI = AQI shifted -1 within station)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _add_lag_features(df: pd.DataFrame, lag_days: list) -> pd.DataFrame:
    """Add AQI lag columns per station."""
    for lag in lag_days:
        col_name = f"AQI_lag{lag}"
        df[col_name] = df.groupby("Station")["AQI"].transform(
            lambda x: x.shift(lag)
        )
        logger.debug(f"  Created {col_name}")
    return df


def _add_rolling_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Add rolling mean / std of AQI per station."""
    for win in windows:
        mean_col = f"AQI_rolling_mean_{win}"
        std_col  = f"AQI_rolling_std_{win}"
        df[mean_col] = df.groupby("Station")["AQI"].transform(
            lambda x: x.shift(1).rolling(win, min_periods=1).mean()
        )
        df[std_col] = df.groupby("Station")["AQI"].transform(
            lambda x: x.shift(1).rolling(win, min_periods=1).std().fillna(0)
        )
        logger.debug(f"  Created {mean_col}, {std_col}")
    return df


def _add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode temporal / seasonal signals."""
    df["Month"]      = df["Date"].dt.month
    df["DayOfYear"]  = df["Date"].dt.dayofyear
    df["Week"]       = df["Date"].dt.isocalendar().week.astype(int)
    df["Is_Weekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # Season (India-centric: monsoon Jun-Sep, winter Dec-Feb, etc.)
    def _season(month):
        if month in [12, 1, 2]:   return 0   # Winter
        if month in [3, 4, 5]:    return 1   # Summer
        if month in [6, 7, 8, 9]: return 2   # Monsoon
        return 3                              # Post-monsoon

    df["Season"] = df["Month"].map(_season)

    # Sine / cosine encoding for cyclical features
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DOW_sin"]   = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DOW_cos"]   = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    return df


def _add_pollutant_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create physically-motivated interaction/ratio features."""
    df["PM_ratio"]      = df["PM2.5"] / (df["PM10"] + 1e-6)
    df["NOx_SO2"]       = df["NO2"] + df["SO2"]
    df["Heat_Humidity"] = df["Temperature"] * df["Humidity"] / 100
    df["Wind_Rain"]     = df["Wind_Speed"] * (df["Rainfall"] + 1)
    df["CO_NO2_ratio"]  = df["CO"] / (df["NO2"] + 1e-6)
    return df


def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Next_day_AQI = AQI shifted by -1 within each station.
    Rows where the next day's AQI is unknown (last record per station)
    will have NaN and must be dropped before training.
    """
    df["Next_day_AQI"] = df.groupby("Station")["AQI"].transform(
        lambda x: x.shift(-1)
    )
    return df


def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df  : cleaned DataFrame
    cfg : project config

    Returns
    -------
    pd.DataFrame with all new features + target column.
                 Rows with any NaN in lag / target columns are dropped.
    """
    logger.info("── Feature engineering ─────────────────────────")

    df = _add_lag_features(df, cfg["features"]["lag_days"])
    df = _add_rolling_features(df, cfg["features"]["rolling_windows"])
    df = _add_seasonal_features(df)
    df = _add_pollutant_interactions(df)
    df = _add_target(df)

    before = len(df)
    df = df.dropna(subset=["Next_day_AQI"] +
                   [f"AQI_lag{l}" for l in cfg["features"]["lag_days"]])
    logger.info(f"Dropped {before - len(df)} rows with NaN in lag/target columns.")
    logger.info(f"Final dataset shape: {df.shape}")

    # Station label-encode (tree models benefit; LR gets dummies separately)
    df["Station_code"] = df["Station"].astype("category").cat.codes

    return df


def get_feature_columns(cfg: dict) -> list:
    """Return the ordered list of feature column names used for modelling."""
    lag_cols     = [f"AQI_lag{l}" for l in cfg["features"]["lag_days"]]
    roll_mean    = [f"AQI_rolling_mean_{w}" for w in cfg["features"]["rolling_windows"]]
    roll_std     = [f"AQI_rolling_std_{w}"  for w in cfg["features"]["rolling_windows"]]
    pollutant    = cfg["features"]["pollutant_cols"]
    weather      = cfg["features"]["weather_cols"]
    seasonal     = ["Month_sin", "Month_cos", "DOW_sin", "DOW_cos",
                    "Season", "Is_Weekend", "DayOfYear", "Week"]
    interaction  = ["PM_ratio", "NOx_SO2", "Heat_Humidity", "Wind_Rain", "CO_NO2_ratio"]
    station      = ["Station_code"]

    return (lag_cols + roll_mean + roll_std + pollutant +
            weather + seasonal + interaction + station)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    from src.data_loader   import load_config, load_raw_data
    from src.preprocessing import clean_data

    cfg   = load_config()
    raw   = load_raw_data(cfg)
    clean = clean_data(raw)
    feat  = engineer_features(clean, cfg)
    cols  = get_feature_columns(cfg)
    print(f"Feature columns ({len(cols)}): {cols}")
    print(feat[cols + ['Next_day_AQI']].head())
