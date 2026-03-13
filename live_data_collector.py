import pandas as pd
import os
import time
import argparse
from datetime import datetime
from fetch_live_data import get_live_data, STATION_MAP

# Configuration
STORAGE_PATH = 'data/live_aqi_dataset.csv'

# The required columns matching historical dataset
COLUMNS = [
    'Date', 'Year', 'Month', 'Day', 'DayOfWeek', 'Station', 'AQI', 
    'AQI_Category', 'PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO', 
    'Temperature', 'Humidity', 'Wind_Speed', 'Rainfall'
]

def get_aqi_category_label(aqi):
    """Categorization matching standard logic."""
    if aqi is None: return "Unknown"
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"

def collect_data():
    """Fetches live AQI from all mapped stations and appends to CSV."""
    print(f"\n--- 📊 Starting Live Data Collection ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---")
    
    new_records = []
    for name in STATION_MAP.keys():
        print(f"Fetching {name}...")
        try:
            live_vals = get_live_data(name)
            if not live_vals:
                continue
            
            now = datetime.now()
            # Calculate a central AQI if API doesn't provide a consolidated one directly in the record
            # Usually WAQI provides 'aqi' as the dominant pollutant's value
            aqi_val = live_vals.get('PM2.5', 0) # Fallback to PM2.5 as primary driver
            
            record = {
                'Date': now.strftime('%Y-%m-%d'),
                'Year': now.year,
                'Month': now.month,
                'Day': now.day,
                'DayOfWeek': now.weekday(),
                'Station': name,
                'AQI': aqi_val,
                'AQI_Category': get_aqi_category_label(aqi_val),
                'PM2.5': live_vals.get('PM2.5'),
                'PM10': live_vals.get('PM10'),
                'NO2': live_vals.get('NO2'),
                'SO2': live_vals.get('SO2'),
                'O3': live_vals.get('O3'),
                'CO': live_vals.get('CO'),
                'Temperature': live_vals.get('Temperature'),
                'Humidity': live_vals.get('Humidity'),
                'Wind_Speed': live_vals.get('Wind_Speed'),
                'Rainfall': live_vals.get('Rainfall', 0.0)
            }
            new_records.append(record)
        except Exception as e:
            print(f"  Error fetching {name}: {e}")
    
    if not new_records:
        print("No new data fetched.")
        return

    df_new = pd.DataFrame(new_records)
    
    # Load existing or create new
    if os.path.exists(STORAGE_PATH) and os.path.getsize(STORAGE_PATH) > 0:
        try:
            df_existing = pd.read_csv(STORAGE_PATH)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined['Date'] = df_combined['Date'].astype(str)
            
            # Deduplication: Keep the latest fetch for that Station + Date
            df_combined = df_combined.drop_duplicates(subset=['Station', 'Date'], keep='last')
            
            added_count = len(df_combined) - len(df_existing)
            print(f"Collection complete. Added {added_count} new unique records.")
            print(f"Total entries in dataset: {len(df_combined)}")
        except Exception as e:
            print(f"Error merging data: {e}. Starting fresh.")
            df_combined = df_new
    else:
        df_combined = df_new
        if not os.path.exists('data'): os.makedirs('data')
        print(f"Created new dataset with {len(df_new)} records.")

    # Save
    df_combined[COLUMNS].to_csv(STORAGE_PATH, index=False)
    print(f"Saved to {STORAGE_PATH}")

def start_automated_collection(interval_hours=24):
    """Runs the collection loop indefinitely."""
    print(f"🚀 Automated data collection active. Interval: {interval_hours} hours.")
    print("Press Ctrl+C to terminate the process.")
    
    try:
        while True:
            collect_data()
            print(f"Sleeping for {interval_hours} hours...")
            time.sleep(interval_hours * 3600)
    except KeyboardInterrupt:
        print("\nStopping automated collection system...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQI Live Data Collector")
    parser.add_argument('--mode', choices=['once', 'daily', 'hourly'], default='once', 
                        help="Run mode: 'once' (default), 'daily' (24h loop), or 'hourly' (1h loop)")
    args = parser.parse_args()

    if args.mode == 'once':
        collect_data()
    elif args.mode == 'daily':
        start_automated_collection(24)
    elif args.mode == 'hourly':
        start_automated_collection(1)

