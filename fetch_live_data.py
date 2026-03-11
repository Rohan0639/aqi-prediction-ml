import requests
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AQI_API_KEY")
BASE_URL = "https://api.waqi.info/feed/{}/?token={}"

# Accurate mapping of Model Stations to WAQI API IDs
STATION_MAP = {
    "Balanagar SPCB": "@8179",
    "HITEC City": "@9129",
    "IDA Pashamylaram SPCB": "@9144",
    "Sanathnagar SPCB": "@8182",
    "US Consulate": "@7022",
    "Uppal SPCB": "@11333",
    "Zoo Park SPCB": "@8677"
}

def get_live_data(station_name="@8182"):
    """
    Fetch live data from WAQI API for a specific station.
    """
    if not API_KEY:
        raise ValueError("AQI_API_KEY not found in .env file")

    # If a friendly name is passed, convert to ID
    waqi_id = STATION_MAP.get(station_name, station_name)

    url = BASE_URL.format(waqi_id, API_KEY)
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return None

    data = response.json()
    if data['status'] != 'ok':
        print(f"API Error: {data['data']}")
        return None

    iaqi = data['data']['iaqi']
    
    # Extract available pollutants and weather data with high accuracy
    # WAQI: pm25, pm10, no2, so2, o3, co, t (temp), h (humidity), w (wind)
    live_record = {
        'PM2.5': float(iaqi.get('pm25', {}).get('v', 0)),
        'PM10': float(iaqi.get('pm10', {}).get('v', 0)),
        'NO2': float(iaqi.get('no2', {}).get('v', 0)),
        'SO2': float(iaqi.get('so2', {}).get('v', 0)),
        'O3': float(iaqi.get('o3', {}).get('v', 0)),
        'CO': float(iaqi.get('co', {}).get('v', 0)),
        'Temperature': float(iaqi.get('t', {}).get('v', 25.0)),
        'Humidity': float(iaqi.get('h', {}).get('v', 50.0)),
        'Wind_Speed': float(iaqi.get('w', {}).get('v', 5.0)),
        'Rainfall': 0.0 # Rainfall is not systematically provided by WAQI
    }
    
    return live_record

if __name__ == "__main__":
    # Test fetch for all mapped stations
    print("--- Verifying Live Data for All Model Stations ---")
    for name, wid in STATION_MAP.items():
        print(f"\nStation: {name} (ID: {wid})")
        data = get_live_data(name)
        if data:
            print(f"  Status: OK | PM2.5: {data['PM2.5']} | Temp: {data['Temperature']}")
        else:
            print(f"  Status: FAILED")
