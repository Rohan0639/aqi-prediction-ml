"""
test_api_connection.py
======================
Simple standalone script to verify real-time AQI data collection from the WAQI API.
It checks if the fetched data corresponds to today's date and shows the values.
"""

import os
import sys
import yaml
import requests
from datetime import datetime

# Add project root to path so we can import from src
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def test_api():
    config_path = os.path.join(ROOT_DIR, "config", "config.yaml")

    print("=" * 60)
    print("  🔍 AQI API CONNECTION TEST")
    print("=" * 60)

    # 1. Load Config
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found at {config_path}")
        return

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    token = cfg["api"]["waqi_token"]
    city  = cfg["api"]["default_city"]

    print(f"📡 API Token : {token[:5]}...{token[-5:]}")
    print(f"📍 Target City: {city}")

    # 2. Test Connection
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    print(f"🌐 Requesting: {url.split('?')[0]} (token hidden)")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            print(f"❌ API Error: {data.get('data')}")
            return

        # 3. Verify Today's Date
        obs_data = data["data"]
        aqi      = obs_data.get("aqi")
        station  = obs_data.get("city", {}).get("name")
        time_utc = obs_data.get("time", {}).get("s") # Last update time

        today = datetime.now().strftime("%Y-%m-%d")

        print("\n✅ API Response Received!")
        print("-" * 40)
        print(f"🏙️  Station     : {station}")
        print(f"📅  Observation Time: {time_utc}")
        print(f"📊  Current AQI  : {aqi}")
        print("-" * 40)

        if today in time_utc:
            print(f"✨ SUCCESS: The data is from today ({today})!")
        else:
            print(f"⚠️  WARNING: Observation time ({time_utc}) does not match local date ({today}).")
            print("   (This might be due to UTC timezone differences or late station updates.)")

    except Exception as e:
        print(f"💥 Connection Failed: {e}")

if __name__ == "__main__":
    test_api()
