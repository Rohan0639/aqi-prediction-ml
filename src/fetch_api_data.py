"""
fetch_api_data.py
=================
Fetches real-time AQI and environmental data from the WAQI API
(https://waqi.info) and converts it into the same feature-row format
used during model training, so it can be passed directly to predict.py.
"""

import logging
from datetime import date, datetime, timedelta

import numpy as np
import requests
import yaml

logger = logging.getLogger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_cfg(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_station_data(station_uid: str, token: str) -> dict | None:
    """Fetch current observation for one station UID."""
    url = f"https://api.waqi.info/feed/{station_uid}/?token={token}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "ok":
            return data["data"]
    except requests.RequestException as exc:
        logger.warning(f"  Request failed for {station_uid}: {exc}")
    return None


def _extract_iaqi(iaqi: dict, key: str, default=np.nan) -> float:
    """Safely extract a value from the WAQI 'iaqi' dict."""
    entry = iaqi.get(key, {})
    return float(entry.get("v", default)) if entry else default


def _build_feature_row(station_data: dict, station_name: str,
                       station_code: int, cfg: dict) -> dict | None:
    """
    Convert a WAQI station observation JSON to a model-ready feature dict.

    The returned dict has the same field names used during training so that
    feature_engineering.engineer_features (and its lag columns) can be
    re-created directly for a single-row prediction.
    """
    try:
        iaqi = station_data.get("iaqi", {})
        aqi  = float(station_data.get("aqi", np.nan))
        if np.isnan(aqi):
            return None

        t  = datetime.now()
        row = {
            "Date":        t.date(),
            "Year":        t.year,
            "Month":       t.month,
            "Day":         t.day,
            "DayOfWeek":   t.weekday(),
            "Station":     station_name,
            "Station_code": station_code,
            "AQI":         aqi,
            # Pollutants (fall back to NaN if not in API)
            "PM2.5":       _extract_iaqi(iaqi, "pm25"),
            "PM10":        _extract_iaqi(iaqi, "pm10"),
            "NO2":         _extract_iaqi(iaqi, "no2"),
            "SO2":         _extract_iaqi(iaqi, "so2"),
            "O3":          _extract_iaqi(iaqi, "o3"),
            "CO":          _extract_iaqi(iaqi, "co"),
            # Weather
            "Temperature": _extract_iaqi(iaqi, "t"),
            "Humidity":    _extract_iaqi(iaqi, "h"),
            "Wind_Speed":  _extract_iaqi(iaqi, "w"),
            "Rainfall":    _extract_iaqi(iaqi, "r", 0.0),
        }
        return row
    except Exception as exc:
        logger.warning(f"  Failed to parse station data: {exc}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_all_stations_data(config_path: str = "config/config.yaml") -> list:
    """
    Query the WAQI API for each configured station and return a list
    of all successful observations.
    """
    cfg     = _load_cfg(config_path)
    token   = cfg["api"]["waqi_token"]
    stations = cfg["api"]["stations"]

    logger.info(f"Querying WAQI API for {len(stations)} stations…")
    results = []

    for i, uid in enumerate(stations):
        logger.info(f"  [{i+1}/{len(stations)}] Station UID: {uid}")
        data = _get_station_data(uid, token)
        if data is not None:
            station_name = data.get("city", {}).get("name", f"Station_{i}")
            
            # Safety Filter: Ensure station is in Hyderabad or India
            url_str = data.get("city", {}).get("url", "").lower()
            if "hyderabad" not in station_name.lower() and "india" not in station_name.lower() and "hyderabad" not in url_str:
                logger.warning(f"    ⚠️  Skipping Station '{station_name}' (Location mismatch: Not Hyderabad/India)")
                continue

            feature_row = _build_feature_row(data, station_name, i, cfg)
            if feature_row is not None:
                logger.info(f"    ✅ Got data for '{station_name}'")
                results.append(feature_row)
            else:
                logger.warning(f"    ⚠️  Failed to parse data for UID: {uid}")
        else:
            logger.warning(f"    ❌ No data returned for UID: {uid}")

    return results


def fetch_live_data(config_path: str = "config/config.yaml") -> dict | None:
    """
    Query the WAQI API for each configured station and return the first
    successful observation as an enriched feature dict.

    Returns
    -------
    dict  – feature row with all raw columns (no lag/rolling yet)
    None  – if all stations fail
    """
    results = fetch_all_stations_data(config_path)
    if results:
        return results[0]

    # Fallback: try city name directly
    cfg     = _load_cfg(config_path)
    token   = cfg["api"]["waqi_token"]
    city    = cfg["api"]["default_city"]

    logger.info(f"  Trying city-name fallback endpoint: {city}")
    data = _get_station_data(city, token)
    if data:
        station_name = data.get("city", {}).get("name", city)
        feature_row  = _build_feature_row(data, station_name, 0, cfg)
        if feature_row:
            logger.info(f"  Got fallback live data for '{station_name}'")
            return feature_row

    logger.error("All API endpoints failed. Cannot fetch live data.")
    return None


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    
    print("\n--- Testing fetch_all_stations_data ---")
    results = fetch_all_stations_data()
    print(f"\nFetched {len(results)} stations.")
    if results:
        for r in results:
            print(f"- {r['Station']}: AQI={r['AQI']}")
    
    print("\n--- Testing fetch_live_data (first success) ---")
    row = fetch_live_data()
    if row:
        print(f"Single success: {row['Station']} (AQI={row['AQI']})")
