"""
api.py  –  Flask backend for the AQI Prediction Dashboard
==========================================================
Now powered by the 3-model system from train_models.py:
  • Random Forest
  • XGBoost
  • LightGBM
  • Weighted Ensemble (RF×0.4 + LGB×0.4 + XGB×0.2)

Endpoints
---------
GET /api/metrics          – RMSE, MAE, R² for all 4 models
GET /api/predictions      – actual vs predicted (test set)
GET /api/trend            – daily mean AQI trend
GET /api/forecast         – next-day AQI prediction (all models)
GET /api/feature_importance – model feature importances
GET /api/station_stats    – per-station AQI summary
GET /api/aqi_distribution – AQI category distribution
GET /api/monthly_aqi      – monthly seasonal pattern
GET /api/health           – health check

Run with:
    python dashboard/backend/api.py
"""

import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import joblib
import numpy as np
import pandas as pd
import yaml
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Import the same feature pipeline used in training ────────
from train_models import load_data, clean_data, engineer_features, split_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load config ───────────────────────────────────────────────
with open("config/config.yaml") as f:
    CFG = yaml.safe_load(f)

# ── Load the 3 new models + feature list ─────────────────────
logger.info("Loading models from models/…")
RF_MODEL  = joblib.load("models/rf_model.pkl")
XGB_MODEL = joblib.load("models/xgb_model.pkl")
LGB_MODEL = joblib.load("models/lgb_model.pkl")
FEATURE_COLS = joblib.load("models/feature_cols_3model.pkl")
logger.info("Models loaded.")

# ── Build processed dataset using the training pipeline ──────
logger.info("Building feature dataset (this takes ~30s on first run)…")
_raw   = load_data(CFG["paths"]["raw_data"])
_clean = clean_data(_raw)
_feat  = engineer_features(_clean)

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, _FC = split_data(_feat)

# Use the feature list from the saved pickle (guarantees column order matches)
X_TEST_FINAL = _feat.sort_values("Date").reset_index(drop=True).iloc[
    int(len(_feat) * 0.8):
][FEATURE_COLS].values

Y_TEST_FINAL = _feat.sort_values("Date").reset_index(drop=True).iloc[
    int(len(_feat) * 0.8):
]["AQI"].values

TEST_DF = _feat.sort_values("Date").reset_index(drop=True).iloc[
    int(len(_feat) * 0.8):
].copy()

# Pre-compute predictions
RF_PREDS  = RF_MODEL.predict(X_TEST_FINAL).clip(0, 500)
XGB_PREDS = XGB_MODEL.predict(X_TEST_FINAL).clip(0, 500)
LGB_PREDS = LGB_MODEL.predict(X_TEST_FINAL).clip(0, 500)
ENS_PREDS = (0.4 * RF_PREDS + 0.4 * LGB_PREDS + 0.2 * XGB_PREDS).clip(0, 500)

FULL_DF   = _feat.copy()
logger.info("Dataset ready.")

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


def _aqi_cat(v):
    v = float(v)
    if v <= 50:   return "Good"
    if v <= 100:  return "Satisfactory"
    if v <= 200:  return "Moderate"
    if v <= 300:  return "Poor"
    if v <= 400:  return "Very Poor"
    return "Severe"


def _metrics(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {
        "model": name,
        "rmse":  round(rmse, 2),
        "mae":   round(mae,  2),
        "r2":    round(r2,   4),
    }


# ── /api/metrics ──────────────────────────────────────────────
@app.route("/api/metrics")
def metrics():
    return jsonify([
        _metrics(Y_TEST_FINAL, RF_PREDS,  "Random Forest"),
        _metrics(Y_TEST_FINAL, XGB_PREDS, "XGBoost"),
        _metrics(Y_TEST_FINAL, LGB_PREDS, "LightGBM"),
        _metrics(Y_TEST_FINAL, ENS_PREDS, "Ensemble"),
    ])


# ── /api/predictions ─────────────────────────────────────────
@app.route("/api/predictions")
def predictions():
    n    = min(500, len(TEST_DF))
    rows = []
    xp   = RF_PREDS[-n:]
    xgbp = XGB_PREDS[-n:]
    lgbp = LGB_PREDS[-n:]
    ensp = ENS_PREDS[-n:]
    df   = TEST_DF.tail(n)
    for i, (_, row) in enumerate(df.iterrows()):
        rows.append({
            "date":          row["Date"].strftime("%Y-%m-%d"),
            "actual":        round(float(row["AQI"]), 1),
            "random_forest": round(float(xp[i]),   1),
            "xgboost":       round(float(xgbp[i]), 1),
            "lightgbm":      round(float(lgbp[i]), 1),
            "ensemble":      round(float(ensp[i]), 1),
            "station":       str(row["Station"]),
            "category":      _aqi_cat(row["AQI"]),
        })
    return jsonify(rows)


# ── /api/trend ────────────────────────────────────────────────
@app.route("/api/trend")
def trend():
    t = (FULL_DF.groupby("Date")["AQI"]
         .mean().reset_index()
         .rename(columns={"AQI": "mean_aqi"}))
    t["30d_ma"] = t["mean_aqi"].rolling(30, min_periods=1).mean()
    return jsonify([
        {
            "date":     d.strftime("%Y-%m-%d"),
            "mean_aqi": round(float(m), 1),
            "ma30":     round(float(ma), 1),
        }
        for d, m, ma in zip(t["Date"], t["mean_aqi"], t["30d_ma"])
    ])


# ── /api/station_stats ────────────────────────────────────────
@app.route("/api/station_stats")
def station_stats():
    g = (FULL_DF.groupby("Station")["AQI"]
         .agg(mean="mean", median="median", max="max", min="min", count="count")
         .reset_index().round(1))
    return jsonify(g.to_dict(orient="records"))


# ── /api/feature_importance ───────────────────────────────────
@app.route("/api/feature_importance")
def feature_importance():
    """Top 20 features from all three models (averaged importance)."""
    rf_imp  = RF_MODEL.feature_importances_
    xgb_imp = XGB_MODEL.feature_importances_
    lgb_imp = LGB_MODEL.feature_importances_

    # Normalise each to [0,1] then average
    def norm(arr): return arr / (arr.max() + 1e-9)
    avg_imp = (norm(rf_imp) + norm(xgb_imp) + norm(lgb_imp)) / 3.0

    pairs = sorted(zip(FEATURE_COLS, avg_imp), key=lambda x: x[1], reverse=True)[:20]
    return jsonify([
        {"feature": f, "importance": round(float(v), 5),
         "rf":  round(float(norm(rf_imp)[FEATURE_COLS.index(f)]),  5),
         "xgb": round(float(norm(xgb_imp)[FEATURE_COLS.index(f)]),5),
         "lgb": round(float(norm(lgb_imp)[FEATURE_COLS.index(f)]),5),
        }
        for f, v in pairs
    ])


# ── /api/forecast ─────────────────────────────────────────────
@app.route("/api/forecast")
def forecast():
    """Next-day prediction using last row of processed data."""
    last = FULL_DF.sort_values("Date").iloc[-1]
    X = np.array([last.get(c, 0) for c in FEATURE_COLS],
                 dtype=float).reshape(1, -1)
    X = np.nan_to_num(X, nan=0.0)

    rf_p  = float(np.clip(RF_MODEL.predict(X)[0],  0, 500))
    xgb_p = float(np.clip(XGB_MODEL.predict(X)[0], 0, 500))
    lgb_p = float(np.clip(LGB_MODEL.predict(X)[0], 0, 500))
    ens_p = float(np.clip(0.4*rf_p + 0.4*lgb_p + 0.2*xgb_p, 0, 500))

    return jsonify({
        "current_aqi":       round(float(last["AQI"]), 1),
        "current_station":   str(last["Station"]),
        "current_date":      last["Date"].strftime("%Y-%m-%d"),
        "current_category":  _aqi_cat(last["AQI"]),
        "rf_pred":           round(rf_p,  1),
        "xgboost_pred":      round(xgb_p, 1),
        "lgboost_pred":      round(lgb_p, 1),
        "ensemble_pred":     round(ens_p, 1),
        "rf_category":       _aqi_cat(rf_p),
        "xgboost_category":  _aqi_cat(xgb_p),
        "lgboost_category":  _aqi_cat(lgb_p),
        "ensemble_category": _aqi_cat(ens_p),
    })


# ── /api/aqi_distribution ────────────────────────────────────
@app.route("/api/aqi_distribution")
def aqi_distribution():
    cats   = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
    bins   = [0, 50, 100, 200, 300, 400, 500]
    labels = cats
    FULL_DF["cat_bin"] = pd.cut(FULL_DF["AQI"], bins=bins, labels=labels)
    counts = FULL_DF["cat_bin"].value_counts()
    return jsonify([{"category": c, "count": int(counts.get(c, 0))} for c in cats])


# ── /api/monthly_aqi ─────────────────────────────────────────
@app.route("/api/monthly_aqi")
def monthly_aqi():
    g = (FULL_DF.groupby("Month")["AQI"]
         .mean().reset_index().rename(columns={"AQI":"avg_aqi"}))
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    g["month_name"] = g["Month"].apply(lambda m: month_names[int(m)-1])
    return jsonify(g[["month_name","avg_aqi"]]
                   .rename(columns={"avg_aqi":"avg"})
                   .assign(avg=lambda d: d["avg"].round(1))
                   .to_dict(orient="records"))


# ── /api/health ───────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "rows": len(FULL_DF),
                    "models": ["Random Forest","XGBoost","LightGBM","Ensemble"]})


if __name__ == "__main__":
    logger.info("AQI Dashboard API on http://localhost:5000")
    app.run(debug=False, port=5000, host="0.0.0.0")
