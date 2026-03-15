import pandas as pd
import joblib
import os
import sys

# Configuration
MODEL_PATH = 'models/trained_model.pkl'
STATION_MODELS_DIR = 'models/station_models'
LIVE_DATA_PATH = 'data/live_aqi_dataset.csv'
HIST_DATA_PATH = 'data/hyderabad_air_quality_10y_combined_fixed.csv'

def load_payload(station_name=None):
    """Load station-specific or global model payload."""
    if station_name:
        filename = station_name.lower().replace(' ', '_') + '.pkl'
        path = os.path.join(STATION_MODELS_DIR, filename)
        if os.path.exists(path):
            return joblib.load(path)
    
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def run_backtest():
    print("=== AQI Backtest & Today's Prediction (March 15, 2026) ===\n")

    if not os.path.exists(LIVE_DATA_PATH):
        print(f"Error: {LIVE_DATA_PATH} not found.")
        return

    # 1. Load Data
    df_live = pd.read_csv(LIVE_DATA_PATH, parse_dates=['Date'])
    df_hist = pd.read_csv(HIST_DATA_PATH, parse_dates=['Date'])
    
    # 2. Identify Today and Lags
    today = pd.Timestamp('2026-03-15')
    yest_date = pd.Timestamp('2026-03-13') # Missing March 14
    db_yest_date = pd.Timestamp('2026-03-12')

    # Filter data for today
    today_data = df_live[df_live['Date'] == today]
    if today_data.empty:
        print("No data found for today (2026-03-15) in live_aqi_dataset.csv")
        return

    # Combine all for history lookup
    df_all = pd.concat([df_hist, df_live], ignore_index=True).sort_values(['Station', 'Date'])

    results = []

    # Map station names to codes using the category codes from training
    # We'll load the global model to get the mapping
    global_payload = load_payload()
    if not global_payload:
        print("Error: Global model payload not found.")
        return
    
    # Create name -> code mapping from the payload's station_mapping
    # Note: station_mapping in payload is {code: name}
    name_to_code = {name: code for code, name in global_payload['station_mapping'].items()}

    from fetch_live_data import convert_aqi_to_concentration

    for _, row in today_data.iterrows():
        station_name = row['Station']
        station_code = name_to_code.get(station_name)
        
        if station_code is None:
            print(f"Skipping {station_name}: Station code not found in model mapping.")
            continue
        
        # Get historical context
        station_history = df_all[df_all['Station'] == station_name]
        
        # Get Lag 1 (March 13)
        lag1_row = station_history[station_history['Date'] == yest_date]
        # Get Lag 2 (March 12)
        lag2_row = station_history[station_history['Date'] == db_yest_date]
        
        if lag1_row.empty or lag2_row.empty:
            past_data = station_history[station_history['Date'] < today].tail(2)
            if len(past_data) < 2:
                print(f"Skipping {station_name}: Not enough history.")
                continue
            lag1_aqi = past_data.iloc[-1]['AQI']
            lag2_aqi = past_data.iloc[-2]['AQI']
        else:
            lag1_aqi = lag1_row.iloc[0]['AQI']
            lag2_aqi = lag2_row.iloc[0]['AQI']

        # Calculate Rolling 3
        rolling_3 = (row['AQI'] + lag1_aqi + lag2_aqi) / 3

        # Prepare Inputs
        payload = load_payload(station_name)
        model = payload['model']
        features = payload['features']

        # CRITICAL: Convert IAQI to Concentration for the model features
        # The row['PM2.5'] in live_aqi_dataset.csv currently contains IAQI
        pm25_conc = convert_aqi_to_concentration(row['PM2.5'], 'PM2.5')
        pm10_conc = convert_aqi_to_concentration(row['PM10'], 'PM10')

        input_data = {
            'PM2.5': pm25_conc,
            'PM10': pm10_conc,
            'NO2': row['NO2'],
            'SO2': row['SO2'],
            'O3': row['O3'],
            'CO': row['CO'],
            'Temperature': row['Temperature'],
            'Humidity': row['Humidity'],
            'Wind_Speed': row['Wind_Speed'],
            'Rainfall': row['Rainfall'],
            'Station_Code': station_code,
            'AQI_Lag_1': lag1_aqi,
            'AQI_Lag_2': lag2_aqi,
            'AQI_Rolling_3': rolling_3
        }
        
        df_input = pd.DataFrame([input_data])[features]
        prediction = model.predict(df_input)[0]
        
        results.append({
            'Station': station_name,
            'Current AQI': row['AQI'],
            'Lag 1 (Mar 13)': lag1_aqi,
            'Lag 2 (Mar 12)': lag2_aqi,
            'Predicted (Mar 16)': round(prediction, 2)
        })

    # Display Results
    if results:
        res_df = pd.DataFrame(results)
        print(res_df.to_string(index=False))
        print(f"\nSummary: Generated predictions for {len(results)} stations.")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    run_backtest()
