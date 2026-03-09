"""
train_models.py
===============
Complete AQI Next-Day Prediction Training Script
------------------------------------------------
Models trained:
  • Random Forest Regressor
  • XGBoost Regressor
  • LightGBM Regressor
  • Weighted Ensemble (RF 40% + LGB 40% + XGB 20%)

Usage:
    python train_models.py

Outputs:
  - Console: model comparison table + ensemble score
  - models/  rf_model.pkl, xgb_model.pkl, lgb_model.pkl
  - plots/   feature_importance_{model}.png
"""

import os
import warnings

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_PATH   = "data/raw/hyderabad_air_quality_10y_combined_fixed.csv"
MODELS_DIR  = "models"
PLOTS_DIR   = "plots"
TEST_SIZE   = 0.20         # last 20% of rows is test set
RANDOM_SEED = 42

# Lag days and rolling windows to create
LAG_DAYS      = [1, 3, 7, 14, 30]
ROLLING_WINS  = [7, 14, 30]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load CSV, parse dates, and sort chronologically."""
    print(f"\n{'─'*55}")
    print("  [1/9] Loading data…")
    print(f"{'─'*55}")

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"  Rows loaded   : {len(df):,}")
    print(f"  Columns       : {df.shape[1]}")
    print(f"  Date range    : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Stations      : {df['Station'].nunique()} unique")
    return df


# ─────────────────────────────────────────────────────────────
# 2.  DATA CLEANING
# ─────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns, remove duplicates, fill missing values."""
    print(f"\n{'─'*55}")
    print("  [2/9] Cleaning data…")
    print(f"{'─'*55}")

    # Drop columns that leak the target label or are non-numeric identifiers
    drop_cols = ["AQI_Category"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["Date","Station"])
    print(f"  Duplicates removed  : {before - len(df)}")

    # Numeric columns – forward-fill within station, then global median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_before = df[numeric_cols].isnull().sum().sum()
    if missing_before:
        df = df.sort_values(["Station","Date"])
        df[numeric_cols] = (
            df.groupby("Station")[numeric_cols]
            .transform(lambda g: g.ffill().bfill())
        )
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    print(f"  Missing values filled: {missing_before}")
    print(f"  Cleaned shape        : {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag/rolling features and a binary target marker."""
    print(f"\n{'─'*55}")
    print("  [3/9] Engineering features…")
    print(f"{'─'*55}")

    df = df.sort_values(["Station","Date"]).reset_index(drop=True)

    # ── Lag features (shift AQI within each station group)
    for lag in LAG_DAYS:
        col = f"AQI_lag{lag}"
        df[col] = df.groupby("Station")["AQI"].transform(lambda x: x.shift(lag))
        print(f"  Created {col}")

    # ── Rolling mean (using lagged values to avoid data leakage)
    for win in ROLLING_WINS:
        col = f"AQI_rolling_mean_{win}"
        df[col] = df.groupby("Station")["AQI"].transform(
            lambda x: x.shift(1).rolling(win, min_periods=1).mean()
        )
        print(f"  Created {col}")

    # ── Seasonal / calendar features
    df["Month_sin"]  = np.sin(2 * np.pi * df["Date"].dt.month / 12)
    df["Month_cos"]  = np.cos(2 * np.pi * df["Date"].dt.month / 12)
    df["DOW_sin"]    = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DOW_cos"]    = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    df["Is_Weekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Season"]     = df["Date"].dt.month.map(
        lambda m: 0 if m in [12,1,2] else 1 if m in [3,4,5]
                  else 2 if m in [6,7,8,9] else 3
    )
    print("  Created calendar / seasonal features")

    # ── Interaction features
    df["PM_ratio"]       = df["PM2.5"] / (df["PM10"] + 1e-6)
    df["NOx_SO2"]        = df["NO2"] + df["SO2"]
    df["Heat_Humidity"]  = df["Temperature"] * df["Humidity"] / 100
    df["Wind_Rain"]      = df["Wind_Speed"] * (df["Rainfall"] + 1)
    print("  Created interaction features")

    # ── Station label-encoding (for tree models)
    df["Station_code"] = df["Station"].astype("category").cat.codes

    # ── Drop rows where lag features are NaN (first N rows per station)
    lag_cols = [f"AQI_lag{l}" for l in LAG_DAYS]
    before   = len(df)
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    print(f"  Rows after lag-NaN drop: {len(df):,}  (removed {before-len(df)})")

    return df


# ─────────────────────────────────────────────────────────────
# 4.  TRAIN / TEST SPLIT  (time-based, NO shuffle)
# ─────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    """
    Time-based 80/20 split.
    Returns X_train, X_test, y_train, y_test plus the feature column list.
    """
    print(f"\n{'─'*55}")
    print("  [4/9] Train/Test split (time-based 80/20)…")
    print(f"{'─'*55}")

    # Columns NOT used as features
    exclude = {"Date", "Station", "AQI", "AQI_Category",
               "Year", "Month", "Day", "DayOfWeek"}
    feature_cols = [c for c in df.columns if c not in exclude]

    df = df.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - TEST_SIZE))

    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    X_train = train[feature_cols].values
    y_train = train["AQI"].values
    X_test  = test[feature_cols].values
    y_test  = test["AQI"].values

    print(f"  Train rows : {len(train):,}  ({train['Date'].min().date()} → {train['Date'].max().date()})")
    print(f"  Test  rows : {len(test):,}   ({test['Date'].min().date()}  → {test['Date'].max().date()})")
    print(f"  Features   : {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols


# ─────────────────────────────────────────────────────────────
# 5.  MODEL TRAINING
# ─────────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    print("\n  → Training Random Forest…")
    model = RandomForestRegressor(
        n_estimators = 300,
        max_depth    = 12,
        min_samples_leaf = 5,
        n_jobs       = -1,
        random_state = RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    print("    Done.")
    return model


def train_xgboost(X_train, y_train) -> XGBRegressor:
    print("  → Training XGBoost…")
    model = XGBRegressor(
        n_estimators     = 500,
        learning_rate    = 0.05,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
        verbosity        = 0,
    )
    model.fit(X_train, y_train)
    print("    Done.")
    return model


def train_lightgbm(X_train, y_train) -> lgb.LGBMRegressor:
    print("  → Training LightGBM…")
    model = lgb.LGBMRegressor(
        n_estimators  = 500,
        learning_rate = 0.05,
        num_leaves    = 31,
        max_depth     = -1,
        subsample     = 0.8,
        colsample_bytree = 0.8,
        random_state  = RANDOM_SEED,
        n_jobs        = -1,
        verbose       = -1,
    )
    model.fit(X_train, y_train)
    print("    Done.")
    return model


# ─────────────────────────────────────────────────────────────
# 6.  EVALUATION HELPER
# ─────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, name: str) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"Model": name, "RMSE": round(rmse,2), "MAE": round(mae,2), "R²": round(r2,4)}


# ─────────────────────────────────────────────────────────────
# 7.  FEATURE IMPORTANCE PLOTS
# ─────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_cols: list, model_name: str,
                            top_n: int = 20):
    """Save a horizontal bar chart of the top-N feature importances."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        print(f"  Skipping importance plot for {model_name} (not supported).")
        return

    pairs = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)[:top_n]
    names, vals = zip(*pairs)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Blues_r(np.linspace(0.3, 0.85, len(names)))
    ax.barh(range(len(names)), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=10)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"feature_importance_{model_name.lower().replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 8.  ENSEMBLE PREDICTION
# ─────────────────────────────────────────────────────────────
def ensemble_predict(rf_pred, xgb_pred, lgb_pred,
                     w_rf=0.4, w_xgb=0.2, w_lgb=0.4) -> np.ndarray:
    """Weighted average: RF 40% | LGB 40% | XGB 20%."""
    return w_rf * rf_pred + w_lgb * lgb_pred + w_xgb * xgb_pred


# ─────────────────────────────────────────────────────────────
# 9.  MODEL SAVING
# ─────────────────────────────────────────────────────────────
def save_models(rf, xgb, lgbm, feature_cols: list):
    """Persist all three models + feature list using Joblib."""
    joblib.dump(rf,   os.path.join(MODELS_DIR, "rf_model.pkl"))
    joblib.dump(xgb,  os.path.join(MODELS_DIR, "xgb_model.pkl"))
    joblib.dump(lgbm, os.path.join(MODELS_DIR, "lgb_model.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols_3model.pkl"))
    print(f"\n  Models saved to '{MODELS_DIR}/'")
    print("    rf_model.pkl, xgb_model.pkl, lgb_model.pkl")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  AQI NEXT-DAY PREDICTION — 3-MODEL TRAINING SCRIPT")
    print("=" * 55)

    # ── Steps 1-4 ──────────────────────────────────────────
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, feature_cols = split_data(df)

    # ── Step 5: Train ───────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  [5/9] Training Three Models…")
    print(f"{'─'*55}")
    rf_model  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost      (X_train, y_train)
    lgb_model = train_lightgbm     (X_train, y_train)

    # ── Predictions ─────────────────────────────────────────
    rf_pred  = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    lgb_pred = lgb_model.predict(X_test)
    ens_pred = ensemble_predict(rf_pred, xgb_pred, lgb_pred)

    # ── Step 6: Evaluation table ────────────────────────────
    print(f"\n{'─'*55}")
    print("  [6/9] Model Evaluation")
    print(f"{'─'*55}")

    results = [
        evaluate(y_test, rf_pred,  "Random Forest"),
        evaluate(y_test, xgb_pred, "XGBoost"),
        evaluate(y_test, lgb_pred, "LightGBM"),
        evaluate(y_test, ens_pred, "Ensemble (RF+LGB+XGB)"),
    ]

    results_df = pd.DataFrame(results).set_index("Model")
    print()
    print(results_df.to_string())
    print()

    best = results_df["RMSE"].idxmin()
    print(f"  ★ Best model: {best}  "
          f"(RMSE={results_df.loc[best,'RMSE']:.2f},  "
          f"MAE={results_df.loc[best,'MAE']:.2f},  "
          f"R²={results_df.loc[best,'R²']:.4f})")

    # ── Step 7: Feature importances ─────────────────────────
    print(f"\n{'─'*55}")
    print("  [7/9] Feature Importance Plots")
    print(f"{'─'*55}")
    plot_feature_importance(rf_model,  feature_cols, "Random Forest")
    plot_feature_importance(xgb_model, feature_cols, "XGBoost")
    plot_feature_importance(lgb_model, feature_cols, "LightGBM")

    # ── Prediction vs Actual plot ────────────────────────────
    print(f"\n{'─'*55}")
    print("  [8/9] Prediction vs Actual Plots")
    print(f"{'─'*55}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Predicted vs Actual AQI (Test Set)", fontsize=14, fontweight="bold")

    for ax, preds, name in zip(
        axes.flat,
        [rf_pred, xgb_pred, lgb_pred, ens_pred],
        ["Random Forest", "XGBoost", "LightGBM", "Ensemble"]
    ):
        m = evaluate(y_test, preds, name)
        ax.scatter(y_test, preds, alpha=0.3, s=8, color="#4C72B0")
        lim = [min(y_test.min(), preds.min()) - 5,
               max(y_test.max(), preds.max()) + 5]
        ax.plot(lim, lim, "r--", linewidth=1.5)
        ax.set_title(f"{name}\nRMSE={m['RMSE']}  MAE={m['MAE']}  R²={m['R²']}", fontsize=10)
        ax.set_xlabel("Actual AQI", fontsize=9)
        ax.set_ylabel("Predicted AQI", fontsize=9)

    plt.tight_layout()
    scatter_path = os.path.join(PLOTS_DIR, "all_models_predicted_vs_actual.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"  Saved: {scatter_path}")

    # ── Step 9: Save models ──────────────────────────────────
    print(f"\n{'─'*55}")
    print("  [9/9] Saving Models")
    print(f"{'─'*55}")
    save_models(rf_model, xgb_model, lgb_model, feature_cols)

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    print(f"  Models : {MODELS_DIR}/")
    print(f"  Plots  : {PLOTS_DIR}/")
    print()
    print(results_df.to_string())
    print()


if __name__ == "__main__":
    main()
