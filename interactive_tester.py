import pandas as pd
import joblib
import os
import random

# Configuration
DATA_PATH = 'data/hyderabad_air_quality_10y_combined_fixed.csv'
MODEL_PATH = 'models/trained_model.pkl'
STATION_MODELS_DIR = 'models/station_models'

def engineer_features(df):
    """Same logic as training to ensure feature consistency."""
    df = df.sort_values(['Station', 'Date'])
    df['Station_Code'] = df['Station'].astype('category').cat.codes
    df['AQI_Lag_1'] = df.groupby('Station')['AQI'].shift(1)
    df['AQI_Lag_2'] = df.groupby('Station')['AQI'].shift(2)
    df['AQI_Rolling_3'] = df.groupby('Station')['AQI'].transform(lambda x: x.shift(1).rolling(window=3).mean())
    return df.dropna(subset=['AQI_Lag_1', 'AQI_Lag_2', 'AQI_Rolling_3'])

def interactive_test():
    print("--- 🌫️ AQI Model Interactive Tester ---")
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Models not found. Run train_model.py first.")
        return

    # Load Model and Data
    payload = joblib.load(MODEL_PATH)
    model = payload['model']
    features = payload['features']
    station_mapping = payload['station_mapping']

    print("Loading historical data for sampling...")
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    while True:
        print("\nOptions:")
        print("[1] Test on a Random Sample from dataset")
        print("[2] Test on a Specific Station")
        print("[q] Quit")
        
        choice = input("\nSelect an option: ").lower()
        
        if choice == 'q':
            break
        
        test_sample = None
        
        if choice == '1':
            test_sample = df.sample(1).iloc[0]
        elif choice == '2':
            print("\nStations:")
            for code, name in station_mapping.items():
                print(f"  [{code}] {name}")
            try:
                s_code = int(input("Enter Station Code: "))
                station_name = station_mapping[s_code]
                station_df = df[df['Station'] == station_name]
                if len(station_df) > 0:
                    test_sample = station_df.sample(1).iloc[0]
                else:
                    print("No data found for this station.")
                    continue
            except:
                print("Invalid input.")
                continue
        else:
            print("Invalid choice.")
            continue

        if test_sample is not None:
            # Prepare data for prediction
            input_df = pd.DataFrame([test_sample[features]])
            
            # Use station model if available, else global
            station_name = test_sample['Station']
            station_file = os.path.join(STATION_MODELS_DIR, f"{station_name.lower().replace(' ', '_')}.pkl")
            
            current_model = model
            model_type = "Global Model"
            
            if os.path.exists(station_file):
                s_payload = joblib.load(station_file)
                current_model = s_payload['model']
                model_type = f"Station-Specific Model ({station_name})"

            # Predict
            prediction = current_model.predict(input_df)[0]
            actual = test_sample['AQI']
            error = abs(prediction - actual)
            
            print(f"\n--- Test Result ---")
            print(f"Date:           {test_sample['Date']}")
            print(f"Station:        {station_name}")
            print(f"Model Used:     {model_type}")
            print(f"-------------------")
            print(f"Actual AQI:     {actual:.2f}")
            print(f"Predicted AQI:  {prediction:.2f}")
            print(f"Absolute Error: {error:.2f}")
            print(f"-------------------")
            print(f"Current Pollutants: PM2.5={test_sample['PM2.5']:.1f}, PM10={test_sample['PM10']:.1f}, NO2={test_sample['NO2']:.1f}")

if __name__ == "__main__":
    interactive_test()
