"""
train_model.py
==============
Trains Linear Regression and XGBoost models on 80 % of data,
then saves both using Joblib. Also saves the feature column list
and station-to-code mapping for inference time.
"""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def prepare_train_test(df: pd.DataFrame, feature_cols: list, cfg: dict):
    """
    Chronological 80 / 20 split (no shuffle) to avoid data leakage.

    Returns
    -------
    X_train, X_test, y_train, y_test, train_df, test_df
    """
    target = cfg["data"]["target_column"]
    test_size = cfg["data"]["test_size"]

    df = df.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df[target].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target].values

    logger.info(f"Train size: {len(train_df):,}  |  Test size: {len(test_df):,}")
    logger.info(f"Train date range: {train_df['Date'].min().date()} → "
                f"{train_df['Date'].max().date()}")
    logger.info(f"Test  date range: {test_df['Date'].min().date()} → "
                f"{test_df['Date'].max().date()}")
    return X_train, X_test, y_train, y_test, train_df, test_df


def train_linear_regression(X_train, y_train, cfg: dict):
    """Fit Linear Regression on scaled features."""
    logger.info("Training Linear Regression…")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LinearRegression(n_jobs=-1)
    model.fit(X_scaled, y_train)
    logger.info("Linear Regression training complete.")
    return model, scaler


def train_xgboost(X_train, y_train, cfg: dict):
    """Fit XGBoost Regressor with config hyperparameters."""
    xgb_cfg = cfg["models"]["xgboost"]
    logger.info("Training XGBoost…")
    model = XGBRegressor(
        n_estimators     = xgb_cfg["n_estimators"],
        max_depth        = xgb_cfg["max_depth"],
        learning_rate    = xgb_cfg["learning_rate"],
        subsample        = xgb_cfg["subsample"],
        colsample_bytree = xgb_cfg["colsample_bytree"],
        min_child_weight = xgb_cfg["min_child_weight"],
        random_state     = xgb_cfg["random_state"],
        n_jobs           = -1,
        verbosity        = 0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)
    logger.info("XGBoost training complete.")
    return model


def save_models(lr_model, lr_scaler, xgb_model,
                feature_cols: list,
                station_map: dict,
                cfg: dict):
    """Persist models and metadata to disk."""
    out_dir = cfg["paths"]["models_dir"]
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(lr_model,   os.path.join(out_dir, "linear_regression.pkl"))
    joblib.dump(lr_scaler,  os.path.join(out_dir, "lr_scaler.pkl"))
    joblib.dump(xgb_model,  os.path.join(out_dir, "xgboost.pkl"))

    meta = {
        "feature_columns": feature_cols,
        "station_map":     station_map,
        "target":          cfg["data"]["target_column"],
    }
    with open(os.path.join(out_dir, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Models saved to '{out_dir}/'")


def run_training(df: pd.DataFrame, feature_cols: list, cfg: dict):
    """
    Orchestration entry-point called by run_pipeline.py.

    Returns
    -------
    dict with keys: lr_model, lr_scaler, xgb_model,
                    X_test, y_test, test_df
    """
    logger.info("═" * 55)
    logger.info("  MODEL TRAINING")
    logger.info("═" * 55)

    X_train, X_test, y_train, y_test, train_df, test_df = \
        prepare_train_test(df, feature_cols, cfg)

    lr_model, lr_scaler = train_linear_regression(X_train, y_train, cfg)
    xgb_model           = train_xgboost(X_train, y_train, cfg)

    # Station code mapping for inference
    station_map = (
        df[["Station", "Station_code"]]
        .drop_duplicates()
        .set_index("Station")["Station_code"]
        .to_dict()
    )
    # Convert int64 values to plain int for JSON serialisation
    station_map = {k: int(v) for k, v in station_map.items()}

    save_models(lr_model, lr_scaler, xgb_model,
                feature_cols, station_map, cfg)

    return {
        "lr_model":   lr_model,
        "lr_scaler":  lr_scaler,
        "xgb_model":  xgb_model,
        "X_test":     X_test,
        "y_test":     y_test,
        "test_df":    test_df,
    }
