import joblib
import pandas as pd
import os
from fetch_live_data import get_live_data

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

def predict_live():
    # Load global metadata first to get mappings
    global_payload = load_model()
    if not global_payload:
        print(f"Error: No models found. Please run train_model.py first.")
        return

    features = global_payload['features']
    station_mapping = global_payload['station_mapping']

    print("--- Live AQI Prediction ---")
    print("Select a station to fetch live data:")
    for code, name in station_mapping.items():
        print(f"  [{code}] {name}")

    try:
        selection = int(input("\nEnter Station Code: "))
        if selection not in station_mapping:
            print("Invalid station code.")
            return

        station_name = station_mapping[selection]
        
        # Load the best model for this station
        payload = load_model(station_name)
        if not payload:
            print("Failed to load model.")
            return
        model = payload['model']

        print(f"Fetching live data for {station_name}...")
        live_data = get_live_data(station_name)

        if not live_data:
            print(f"Failed to fetch live data for {station_name}.")
            return

        # Attempt to get historical context from local data for Lags and Rolling
        print(f"Retrieving historical context for {station_name}...")
        try:
            # 1. Load both historical and live datasets
            hist_file = 'data/hyderabad_air_quality_10y_combined_fixed.csv'
            live_file = 'data/live_aqi_dataset.csv'
            
            df_hist = pd.read_csv(hist_file, parse_dates=['Date'])
            
            if os.path.exists(live_file):
                df_live = pd.read_csv(live_file, parse_dates=['Date'])
                # Concatenate both datasets
                df_all = pd.concat([df_hist, df_live], ignore_index=True)
            else:
                df_all = df_hist

            # 2. Preprocess to ensure unique Date/Station pairs (keeping the last)
            # This ensures if live data has newer values for the same day, they take precedence
            df_all = df_all.drop_duplicates(subset=['Date', 'Station'], keep='last')
            
            # 3. Filter for the current station and sort
            df_station = df_all[df_all['Station'] == station_name].sort_values('Date')
            
            if len(df_station) >= 3:
                last_3 = df_station.tail(3)['AQI'].tolist()
                # last_3 is [aqi_3d_back, aqi_db_yest, aqi_yest]
                aqi_yest = last_3[2]
                aqi_db_yest = last_3[1]
                aqi_3d_back = last_3[0]
                print(f"  Using historical records: Yesterday={aqi_yest:.1f}, Day Before={aqi_db_yest:.1f}")
            else:
                raise ValueError("Not enough historical data available (need last 3 records).")
        except Exception as e:
            print(f"  Could not load historical context: {e}")
            print("  Please provide context manually:")
            aqi_yest = float(input("    Enter AQI from Yesterday: "))
            aqi_db_yest = float(input("    Enter AQI from Day Before: "))
            aqi_3d_back = float(input("    Enter AQI from 3 Days Ago: "))

        # Calculate Rolling 3
        rolling_3 = (aqi_yest + aqi_db_yest + aqi_3d_back) / 3

        # Prepare DataFrame for prediction (must match model features)
        live_data['Station_Code'] = selection
        live_data['AQI_Lag_1'] = aqi_yest
        live_data['AQI_Lag_2'] = aqi_db_yest
        live_data['AQI_Rolling_3'] = rolling_3
        
        # Ensure correct feature order
        df_input = pd.DataFrame([live_data])[features]

        # Run Prediction
        prediction = model.predict(df_input)[0]
        print(f"\n--- Results ---")
        print(f"Station: {station_name}")
        print(f"Model Used: {'Station-specific' if 'station_name' in payload else 'Global Fallback'}")
        print(f"Predicted Next-Day AQI: {prediction:.2f}")

    except ValueError:
        print("Invalid input. Please enter a valid station code.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    predict_live()
