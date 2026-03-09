"""
evaluate_model.py
=================
Computes RMSE, MAE, R² for both models, prints a comparison table,
declares the best model, and generates three plots:
  1. AQI trend over time
  2. Feature correlation heatmap
  3. Predicted vs Actual (both models)
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# ── Plotting style ────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
PLOT_DIR = "data/processed"


def _compute_metrics(y_true, y_pred, model_name: str) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"Model": model_name, "RMSE": rmse, "MAE": mae, "R²": r2}


def evaluate_models(artifacts: dict, df_full: pd.DataFrame,
                    feature_cols: list, cfg: dict) -> str:
    """
    Evaluate both models, print comparison, generate plots.

    Parameters
    ----------
    artifacts   : dict returned by run_training()
    df_full     : fully-featured DataFrame (used for trend plot)
    feature_cols: list of feature column names
    cfg         : project config

    Returns
    -------
    str : 'linear_regression' or 'xgboost'  (best model name)
    """
    logger.info("═" * 55)
    logger.info("  MODEL EVALUATION")
    logger.info("═" * 55)

    lr_model  = artifacts["lr_model"]
    lr_scaler = artifacts["lr_scaler"]
    xgb_model = artifacts["xgb_model"]
    X_test    = artifacts["X_test"]
    y_test    = artifacts["y_test"]
    test_df   = artifacts["test_df"]

    # ── Predictions ───────────────────────────────────────────
    X_test_scaled = lr_scaler.transform(X_test)
    lr_preds  = lr_model.predict(X_test_scaled)
    xgb_preds = xgb_model.predict(X_test)

    # ── Metrics table ─────────────────────────────────────────
    rows = [
        _compute_metrics(y_test, lr_preds,  "Linear Regression"),
        _compute_metrics(y_test, xgb_preds, "XGBoost"),
    ]
    results_df = pd.DataFrame(rows).set_index("Model")
    print("\n" + "═" * 50)
    print("   MODEL COMPARISON RESULTS")
    print("═" * 50)
    print(results_df.round(4).to_string())
    print("═" * 50)

    # ── Select best model ─────────────────────────────────────
    best_name = results_df["RMSE"].idxmin()
    logger.info(f"Best model: {best_name} "
                f"(RMSE={results_df.loc[best_name,'RMSE']:.2f})")

    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── Plot 1: AQI trend over time (all stations average) ────
    logger.info("Generating AQI trend plot…")
    trend = (df_full.groupby("Date")["AQI"]
             .mean()
             .reset_index()
             .rename(columns={"AQI": "Mean_AQI"}))
    trend["30d_MA"] = trend["Mean_AQI"].rolling(30, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(trend["Date"], trend["Mean_AQI"], alpha=0.35, linewidth=0.8,
            color="#5B9BD5", label="Daily mean AQI")
    ax.plot(trend["Date"], trend["30d_MA"], linewidth=2,
            color="#E05C5C", label="30-day moving avg")
    ax.set_title("Hyderabad AQI Trend (all stations)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "aqi_trend.png"), dpi=150)
    plt.close()
    logger.info("  Saved: data/processed/aqi_trend.png")

    # ── Plot 2: Correlation heatmap ───────────────────────────
    logger.info("Generating correlation heatmap…")
    heatmap_cols = (
        ["AQI", "PM2.5", "PM10", "NO2", "SO2", "O3", "CO",
         "Temperature", "Humidity", "Wind_Speed", "Rainfall",
         "AQI_lag1", "AQI_rolling_mean_7", "Next_day_AQI"]
    )
    heatmap_cols = [c for c in heatmap_cols if c in df_full.columns]
    corr = df_full[heatmap_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"fontsize": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    logger.info("  Saved: data/processed/correlation_heatmap.png")

    # ── Plot 3: Predicted vs Actual ───────────────────────────
    logger.info("Generating prediction vs actual plot…")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, preds, title in zip(
        axes,
        [lr_preds, xgb_preds],
        ["Linear Regression", "XGBoost"]
    ):
        ax.scatter(y_test, preds, alpha=0.35, s=10, color="#5B9BD5")
        lim = [min(y_test.min(), preds.min()) - 5,
               max(y_test.max(), preds.max()) + 5]
        ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect fit")
        metrics = _compute_metrics(y_test, preds, title)
        ax.set_title(f"{title}\nRMSE={metrics['RMSE']:.1f}  "
                     f"MAE={metrics['MAE']:.1f}  R²={metrics['R²']:.3f}",
                     fontsize=11)
        ax.set_xlabel("Actual AQI")
        ax.set_ylabel("Predicted AQI")
        ax.legend()

    plt.suptitle("Predicted vs Actual Next-Day AQI", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "prediction_vs_actual.png"), dpi=150)
    plt.close()
    logger.info("  Saved: data/processed/prediction_vs_actual.png")

    return best_name
