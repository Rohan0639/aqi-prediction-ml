import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Load API Key
load_dotenv()
API_KEY = os.getenv('AQI_API_KEY')
BASE_URL = "https://api.waqi.info/feed/"

# Configuration
STORAGE_PATH = 'data/live_aqi_dataset.csv'
STATIONS = {
    "Balanagar SPCB": "@8179",
    "HITEC City": "@9129",
    "IDA Pashamylaram SPCB": "@9144",
    "Sanathnagar SPCB": "@8182",
    "US Consulate": "@7022",
    "Uppal SPCB": "@11333",
    "Zoo Park SPCB": "@8677"
}

# The required columns matching historical dataset
COLUMNS = [
    'Date', 'Year', 'Month', 'Day', 'DayOfWeek', 'Station', 'AQI', 
    'AQI_Category', 'PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO', 
    'Temperature', 'Humidity', 'Wind_Speed', 'Rainfall'
]

def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"

def fetch_station_data(station_name, waqi_id):
    try:
        url = f"{BASE_URL}{waqi_id}/?token={API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if data['status'] != 'ok':
            print(f"Error fetching {station_name}: {data.get('data')}")
            return None

        content = data['data']
        iaqi = content.get('iaqi', {})
        
        # We use System Time for the logged date to ensure accuracy for the live dataset
        now = datetime.now()
        
        # Build record matching historical schema exactly
        record = {
            'Date': now.strftime('%Y-%m-%d'),
            'Year': now.year,
            'Month': now.month,
            'Day': now.day,
            'DayOfWeek': now.weekday(),
            'Station': station_name,
            'AQI': content.get('aqi'),
            'AQI_Category': get_aqi_category(content.get('aqi')),
            'PM2.5': iaqi.get('pm25', {}).get('v'),
            'PM10': iaqi.get('pm10', {}).get('v'),
            'NO2': iaqi.get('no2', {}).get('v'),
            'SO2': iaqi.get('so2', {}).get('v'),
            'O3': iaqi.get('o3', {}).get('v'),
            'CO': iaqi.get('co', {}).get('v'),
            'Temperature': iaqi.get('t', {}).get('v'),
            'Humidity': iaqi.get('h', {}).get('v'),
            'Wind_Speed': iaqi.get('w', {}).get('v'),
            'Rainfall': 0.0  # WAQI often doesn't provide rainfall
        }
        return record
    except Exception as e:
        print(f"Exception for {station_name}: {e}")
        return None

def collect_data():
    if not API_KEY:
        print("Error: AQI_API_KEY not found in .env file.")
        return

    print(f"--- 📊 Starting Live Data Collection ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---")
    
    new_records = []
    for name, uid in STATIONS.items():
        print(f"Fetching {name}...")
        res = fetch_station_data(name, uid)
        if res:
            new_records.append(res)
    
    if not new_records:
        print("No new data fetched.")
        return

    df_new = pd.DataFrame(new_records)
    
    # Load existing or create new
    if os.path.exists(STORAGE_PATH) and os.path.getsize(STORAGE_PATH) > 0:
        try:
            df_existing = pd.read_csv(STORAGE_PATH)
            
            # Combine
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            
            # Convert Date to string for consistent comparison
            df_combined['Date'] = df_combined['Date'].astype(str)
            
            # Deduplication: Keep the latest fetch for that Station + Date
            df_combined = df_combined.drop_duplicates(subset=['Station', 'Date'], keep='last')
            
            added_count = len(df_combined) - len(df_existing)
            print(f"Collection complete. Total unique records in dataset: {len(df_combined)}")
        except pd.errors.EmptyDataError:
            df_combined = df_new
            print(f"File was empty or corrupted. Reinitialized with {len(df_new)} records.")
    else:
        df_combined = df_new
        print(f"Created new dataset with {len(df_new)} records.")

    # Save
    df_combined[COLUMNS].to_csv(STORAGE_PATH, index=False)
    print(f"Data saved to {STORAGE_PATH}")

if __name__ == "__main__":
    collect_data()
