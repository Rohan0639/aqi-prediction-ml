"""
predict.py
==========
Load the best-trained model, fetch live AQI data from the WAQI API,
engineer features (using historical data for lag/rolling), and predict
next-day AQI.

Usage
-----
Run from the project root:
    python src/predict.py
    python src/predict.py --station "Somajiguda"
    python src/predict.py --offline   # uses last row of processed data
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

# ── Add root to path so we can import 'src' ──────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

logger = logging.getLogger(__name__)


# ── AQI Category helper ───────────────────────────────────────────────────────

def aqi_category(value: float) -> str:
    """Return India NAQI category string for an AQI value."""
    v = float(value)
    if v <= 50:    return "Good"
    if v <= 100:   return "Satisfactory"
    if v <= 200:   return "Moderate"
    if v <= 300:   return "Poor"
    if v <= 400:   return "Very Poor"
    return "Severe"


# ── Feature builder from a raw live row ──────────────────────────────────────

def _build_prediction_features(live_row: dict,
                                history_df: pd.DataFrame,
                                feature_cols: list,
                                station_map: dict,
                                cfg: dict) -> np.ndarray:
    """
    Given a live raw row (from the API) and the historical processed DataFrame,
    reconstruct the full feature vector for prediction.

    Strategy:
    - Lag / rolling features are sourced from the most recent historical rows
      of the same (or nearest) station.
    - Current pollutants / weather come from the live API row.
    """
    station = live_row.get("Station", "unknown")
    t = datetime.now()

    # Find historical rows for this station (most recent first)
    station_hist = history_df[history_df["Station"] == station].sort_values("Date")

    # If station not in history, fall back to any station
    if len(station_hist) == 0:
        station_hist = history_df.sort_values("Date")
        logger.warning(f"Station '{station}' not in history – using global history.")

    last_rows = station_hist.tail(14)  # last 14 days

    def _hist_aqi(shift: int) -> float:
        """AQI from `shift` days ago."""
        if len(last_rows) >= shift:
            return float(last_rows.iloc[-shift]["AQI"])
        return float(last_rows.iloc[0]["AQI"])

    def _roll_mean(window: int) -> float:
        vals = last_rows["AQI"].tail(window)
        return float(vals.mean()) if len(vals) > 0 else float(last_rows["AQI"].mean())

    def _roll_std(window: int) -> float:
        vals = last_rows["AQI"].tail(window)
        return float(vals.std()) if len(vals) > 1 else 0.0

    lag_days = cfg["features"]["lag_days"]
    roll_wins = cfg["features"]["rolling_windows"]

    sc = station_map.get(station, 0)
    month  = t.month
    dow    = t.weekday()

    def _season(m):
        if m in [12,1,2]:   return 0
        if m in [3,4,5]:    return 1
        if m in [6,7,8,9]:  return 2
        return 3

    # Local variables for current observation
    obs = {
        "pm25": live_row.get("PM2.5",  np.nan),
        "pm10": live_row.get("PM10",   np.nan),
        "no2":  live_row.get("NO2",    np.nan),
        "so2":  live_row.get("SO2",    np.nan),
        "o3":   live_row.get("O3",     np.nan),
        "co":   live_row.get("CO",     np.nan),
        "temp": live_row.get("Temperature", np.nan),
        "hum":  live_row.get("Humidity",    np.nan),
        "wind": live_row.get("Wind_Speed",  np.nan),
        "rain": live_row.get("Rainfall",    0.0),
    }

    # Fill NaN sensor readings from historical medians for the same station
    for col_name, var_name in [("PM2.5", "pm25"), ("PM10", "pm10"),
                                ("NO2", "no2"), ("SO2", "so2"),
                                ("O3", "o3"), ("CO", "co"),
                                ("Temperature", "temp"), ("Humidity", "hum"),
                                ("Wind_Speed", "wind"), ("Rainfall", "rain")]:
        if np.isnan(obs[var_name]) and col_name in station_hist.columns:
            filled = float(station_hist[col_name].median())
            obs[var_name] = filled if not np.isnan(filled) else 0.0

    pm25, pm10 = obs["pm25"], obs["pm10"]
    no2, so2   = obs["no2"],  obs["so2"]
    o3, co     = obs["o3"],   obs["co"]
    temp, hum  = obs["temp"], obs["hum"]
    wind, rain = obs["wind"], obs["rain"]

    row_dict: dict = {}
    for lag in lag_days:
        row_dict[f"AQI_lag{lag}"] = _hist_aqi(lag)
    for win in roll_wins:
        row_dict[f"AQI_rolling_mean_{win}"] = _roll_mean(win)
        row_dict[f"AQI_rolling_std_{win}"]  = _roll_std(win)
    for col in cfg["features"]["pollutant_cols"]:
        row_dict[col] = live_row.get(col, np.nan)
    for col in cfg["features"]["weather_cols"]:
        row_dict[col] = live_row.get(col, np.nan)

    row_dict["Month_sin"]   = np.sin(2 * np.pi * month / 12)
    row_dict["Month_cos"]   = np.cos(2 * np.pi * month / 12)
    row_dict["DOW_sin"]     = np.sin(2 * np.pi * dow / 7)
    row_dict["DOW_cos"]     = np.cos(2 * np.pi * dow / 7)
    row_dict["Season"]      = _season(month)
    row_dict["Is_Weekend"]  = int(dow >= 5)
    row_dict["DayOfYear"]   = t.timetuple().tm_yday
    row_dict["Week"]        = int(t.isocalendar()[1])
    row_dict["PM_ratio"]    = pm25 / (pm10 + 1e-6)
    row_dict["NOx_SO2"]     = no2 + so2
    row_dict["Heat_Humidity"] = temp * hum / 100 if not (np.isnan(temp) or np.isnan(hum)) else np.nan
    row_dict["Wind_Rain"]   = wind * (rain + 1)
    row_dict["CO_NO2_ratio"] = co / (no2 + 1e-6)
    row_dict["Station_code"] = sc

    # Build vector in correct column order
    X = np.array([row_dict.get(c, np.nan) for c in feature_cols], dtype=float)

    # Impute any remaining NaN with column medians from history
    if np.isnan(X).any():
        for i, col in enumerate(feature_cols):
            if np.isnan(X[i]) and col in history_df.columns:
                X[i] = float(history_df[col].median())
            elif np.isnan(X[i]):
                X[i] = 0.0

    return X.reshape(1, -1)


# ── Main prediction routine ───────────────────────────────────────────────────

def _append_live_data(live_rows: list, history_df: pd.DataFrame, proc_path: str):
    """
    Append new live observations to the processed CSV if they don't already exist
    for the same Station and Date.
    """
    if not live_rows:
        return history_df

    new_df = pd.DataFrame(live_rows)
    new_df["Date"] = pd.to_datetime(new_df["Date"])
    
    # Ensure date format matches for comparison
    history_df["Date"] = pd.to_datetime(history_df["Date"])
    
    # Identify rows that are NOT already in history (by Station and Date)
    # We use a merge with indicator to find unique rows
    combined = pd.merge(new_df, history_df[['Station', 'Date']], 
                        on=['Station', 'Date'], 
                        how='left', indicator=True)
    
    unique_new = new_df[combined['_merge'] == 'left_only'].copy()

    if len(unique_new) > 0:
        logger.info(f"Adding {len(unique_new)} new unique station readings to {proc_path}")
        # Append to the file
        updated_df = pd.concat([history_df, unique_new], ignore_index=True).sort_values(["Station", "Date"])
        updated_df.to_csv(proc_path, index=False)
        return updated_df
    else:
        logger.info("No new unique station readings to add (already in history).")
        return history_df


def predict_next_day(config_path: str = "config/config.yaml",
                     offline: bool = False,
                     save_data: bool = True):
    """Fetch live data, apply best model, print next-day AQI forecast."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    models_dir = cfg["paths"]["models_dir"]
    proc_path  = cfg["paths"]["processed_data"]

    # ── Load models & metadata ────────────────────────────────
    meta_path = os.path.join(models_dir, "model_metadata.json")
    if not os.path.exists(meta_path):
        logger.error("No trained model found. Run pipeline/run_pipeline.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    feature_cols = meta["feature_columns"]
    station_map  = meta["station_map"]

    xgb_model  = joblib.load(os.path.join(models_dir, "xgboost.pkl"))
    lr_model   = joblib.load(os.path.join(models_dir, "linear_regression.pkl"))
    lr_scaler  = joblib.load(os.path.join(models_dir, "lr_scaler.pkl"))

    # ── Load processed history ────────────────────────────────
    if os.path.exists(proc_path):
        history_df = pd.read_csv(proc_path)
        history_df["Date"] = pd.to_datetime(history_df["Date"])
    else:
        logger.error(f"Processed data not found at {proc_path}. "
                     "Run run_pipeline.py first.")
        sys.exit(1)

    # ── Fetch raw rows from all stations ──────────────────────
    live_rows = []
    if not offline:
        try:
            from src.fetch_api_data import fetch_all_stations_data
            live_rows = fetch_all_stations_data(config_path)
        except Exception as exc:
            logger.warning(f"API fetch failed: {exc}")

    # ── Store and Update History ─────────────────────────────
    if not offline and save_data and live_rows:
        history_df = _append_live_data(live_rows, history_df, proc_path)

    # If no live data, use last known state for all stations
    if not live_rows:
        logger.warning("Performing prediction on last known state for each station (offline mode).")
        # Get the latest row for each station
        live_rows = history_df.sort_values("Date").groupby("Station").tail(1).to_dict("records")

    # ── Prediction Loop ──────────────────────────────────────
    print("\n" + "═" * 120)
    print(f"{'Station':<20} | {'AQI':<6} | {'PM2.5':<6} | {'PM10':<6} | {'Temp':<6} | {'Humid':<6} | {'Pred':<6} | {'Category'}")
    print("─" * 120)

    predictions = []
    for row in live_rows:
        station_name = str(row.get("Station", "Unknown"))
        current_aqi  = float(row.get("AQI", 0))
        pm25         = float(row.get("PM2.5", 0))
        pm10         = float(row.get("PM10", 0))
        temp         = float(row.get("Temperature", 0))
        humid        = float(row.get("Humidity", 0))

        # Build features (this uses historical medians for any NaNs)
        X = _build_prediction_features(row, history_df,
                                       feature_cols, station_map, cfg)

        # Predict
        xgb_p = float(np.clip(xgb_model.predict(X)[0], 0, 500))
        cat   = aqi_category(xgb_p)
        
        print(f"{station_name[:20]:<20} | {current_aqi:<6.1f} | {pm25:<6.1f} | {pm10:<6.1f} | {temp:<6.1f} | {humid:<6.1f} | {xgb_p:<6.1f} | {cat}")
        predictions.append({"station": station_name, "prediction": xgb_p})

    print("═" * 120 + "\n")
    return predictions


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Predict tomorrow's AQI for all stations")
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--offline", action="store_true",
                        help="Skip API call; use last known data points")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save live data to historical file")
    args = parser.parse_args()

    predict_next_day(config_path=args.config, offline=args.offline, save_data=not args.no_save)
