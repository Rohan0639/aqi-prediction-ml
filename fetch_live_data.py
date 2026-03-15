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

def convert_aqi_to_concentration(aqi, pollutant):
    """
    Approximate conversion of AQI sub-index to concentration (ug/m3) 
    based on Indian CPCB standards.
    """
    if aqi <= 0: return 0.0
    
    # Breakpoints for PM2.5 (Concentration vs AQI)
    # [0, 30, 60, 90, 120, 250] -> [0, 50, 100, 200, 300, 400]
    if pollutant == 'PM2.5':
        if aqi <= 50: return (aqi * 30) / 50
        if aqi <= 100: return 30 + (aqi - 50) * (60 - 30) / (100 - 50)
        if aqi <= 200: return 60 + (aqi - 100) * (90 - 60) / (200 - 100)
        if aqi <= 300: return 90 + (aqi - 200) * (120 - 90) / (300 - 200)
        return 120 + (aqi - 300) * (250 - 120) / (400 - 300)

    # Breakpoints for PM10
    # [0, 50, 100, 250, 350, 430] -> [0, 50, 100, 200, 300, 400]
    if pollutant == 'PM10':
        if aqi <= 50: return aqi
        if aqi <= 100: return aqi
        if aqi <= 200: return 100 + (aqi - 100) * (250 - 100) / (200 - 100)
        if aqi <= 300: return 250 + (aqi - 200) * (350 - 250) / (300 - 200)
        return 350 + (aqi - 300) * (430 - 350) / (400 - 300)
        
    return aqi # Fallback for others

def get_live_data(station_name="@8182"):
    """
    Fetch live data from WAQI API for a specific station and convert to concentrations.
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
    
    # Extract IAQI values
    pm25_aqi = float(iaqi.get('pm25', {}).get('v', 0))
    pm10_aqi = float(iaqi.get('pm10', {}).get('v', 0))
    
    # Convert IAQI to Concentration (ug/m3) to match training data
    live_record = {
        'AQI': pm25_aqi, # Keep the original IAQI for display
        'PM2.5': convert_aqi_to_concentration(pm25_aqi, 'PM2.5'),
        'PM10': convert_aqi_to_concentration(pm10_aqi, 'PM10'),
        'NO2': float(iaqi.get('no2', {}).get('v', 0)),
        'SO2': float(iaqi.get('so2', {}).get('v', 0)),
        'O3': float(iaqi.get('o3', {}).get('v', 0)),
        'CO': float(iaqi.get('co', {}).get('v', 0)),
        'Temperature': float(iaqi.get('t', {}).get('v', 25.0)),
        'Humidity': float(iaqi.get('h', {}).get('v', 50.0)),
        'Wind_Speed': float(iaqi.get('w', {}).get('v', 5.0)),
        'Rainfall': 0.0 
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
