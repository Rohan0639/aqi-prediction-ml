import joblib
import pandas as pd
import numpy as np
import sys
import os

MODEL_PATH = 'models/trained_model.pkl'
STATION_MODELS_DIR = 'models/station_models'

def load_model(station_name=None):
    """Load station-specific model or fallback to global."""
    if station_name:
        station_file = os.path.join(STATION_MODELS_DIR, f"{station_name.lower().replace(' ', '_')}.pkl")
        if os.path.exists(station_file):
            print(f"Loading station-specific model for: {station_name}")
            return joblib.load(station_file)
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading global fallback model...")
        return joblib.load(MODEL_PATH)
    
    return None

def predict():
    # Load global metadata first to get mappings
    global_payload = load_model()
    if not global_payload:
        print(f"Error: No models found. Please run train_model.py first.")
        return

    features = global_payload['features']
    station_mapping = global_payload['station_mapping']

    print("--- AQI Next-Day Prediction ---")
    print("Available Stations:")
    for code, name in station_mapping.items():
        print(f"{code}: {name}")
    
    try:
        station_code = int(input("\nEnter Station Code: "))
        if station_code not in station_mapping:
            print("Invalid station code.")
            return

        station_name = station_mapping[station_code]
        
        # Load the best model for this station
        payload = load_model(station_name)
        if not payload:
            print("Failed to load model.")
            return
        model = payload['model']

        from fetch_live_data import convert_aqi_to_concentration

        print("\n--- Enter Live Pollutant Indicators (AQI Sub-indices) ---")
        pm25_aqi = float(input("Enter PM2.5 (IAQI): "))
        pm10_aqi = float(input("Enter PM10 (IAQI): "))
        no2 = float(input("Enter NO2: "))
        so2 = float(input("Enter SO2: "))
        o3 = float(input("Enter O3: "))
        co = float(input("Enter CO: "))
        temp = float(input("Enter Temperature: "))
        hum = float(input("Enter Humidity: "))
        wind = float(input("Enter Wind Speed: "))
        rain = float(input("Enter Rainfall: "))

        # Convert IAQI to Concentration for model compatibility
        pm25 = convert_aqi_to_concentration(pm25_aqi, 'PM2.5')
        pm10 = convert_aqi_to_concentration(pm10_aqi, 'PM10')

        print("\n--- Historical Context (Required for Lags & Rolling) ---")
        aqi_yest = float(input("Enter AQI from Yesterday (Lag 1): "))
        aqi_db_yest = float(input("Enter AQI from Day Before (Lag 2): "))
        aqi_3d_back = float(input("Enter AQI from 3 Days Ago: "))

        # Calculate Rolling 3 (Mean of trailing 3 days excluding today)
        rolling_3 = (aqi_yest + aqi_db_yest + aqi_3d_back) / 3

        # Prepare input data
        input_data = pd.DataFrame([[
            pm25, pm10, no2, so2, o3, co, temp, hum, wind, rain, station_code,
            aqi_yest, aqi_db_yest, rolling_3
        ]], columns=features)

        # Predict
        prediction = model.predict(input_data)[0]
        print(f"\n--- Results ---")
        print(f"Station: {station_name}")
        print(f"Model Used: {'Station-specific' if 'station_name' in payload else 'Global Fallback'}")
        print(f"Predicted Next-Day AQI: {prediction:.2f}")

    except ValueError:
        print("Invalid input. Please enter numeric values.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    predict()
