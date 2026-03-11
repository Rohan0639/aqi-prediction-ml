import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_PATH = 'data/hyderabad_air_quality_10y_combined_fixed.csv'
MODEL_PATH = 'models/trained_model.pkl'
STATION_MODELS_DIR = 'models/station_models'
PLOTS_DIR = 'plots/performance_test'
TARGET = 'AQI'

def engineer_features(df):
    """Create lag features matching train_model.py logic."""
    df = df.sort_values(['Station', 'Date'])
    
    # Create Station Code
    df['Station_Code'] = df['Station'].astype('category').cat.codes
    
    # Create Lag Features
    df['AQI_Lag_1'] = df.groupby('Station')['AQI'].shift(1)
    df['AQI_Lag_2'] = df.groupby('Station')['AQI'].shift(2)
    
    # Create Rolling Average
    df['AQI_Rolling_3'] = df.groupby('Station')['AQI'].transform(lambda x: x.shift(1).rolling(window=3).mean())
    
    # Drop rows with NaN values created by shifts
    df = df.dropna(subset=['AQI_Lag_1', 'AQI_Lag_2', 'AQI_Rolling_3', 'PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO', 'Temperature', 'Humidity', 'Wind_Speed', 'Rainfall'])
    
    return df

def evaluate_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n--- Performance Metrics: {name} ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")
    
    return {'Name': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

def plot_results(y_true, y_pred, name):
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, color='forestgreen')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title(f'Actual vs Predicted - {name}')
    
    filename = os.path.join(PLOTS_DIR, f"{name.lower().replace(' ', '_')}_eval.png")
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to: {filename}")

def run_test():
    print("Loading data and model...")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Run train_model.py first.")
        return

    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    df = engineer_features(df)
    
    payload = joblib.load(MODEL_PATH)
    model = payload['model']
    features = payload['features']
    
    # Use the same split as training (25% test, sequential)
    # We sort by date to ensure the test set is the most recent data
    df = df.sort_values('Date')
    split_idx = int(len(df) * 0.75)
    test_df = df.iloc[split_idx:]
    
    X_test = test_df[features]
    y_test = test_df[TARGET]
    
    # 1. Global Model Evaluation
    preds = model.predict(X_test)
    results = [evaluate_model(y_test, preds, 'Global Model')]
    plot_results(y_test, preds, 'Global Model')
    
    # 2. Station-wise Evaluation
    print("\n\n--- Station-wise Evaluation ---")
    stations = test_df['Station'].unique()
    
    for station in stations:
        station_file = os.path.join(STATION_MODELS_DIR, f"{station.lower().replace(' ', '_')}.pkl")
        if os.path.exists(station_file):
            s_payload = joblib.load(station_file)
            s_model = s_payload['model']
            
            s_test_df = test_df[test_df['Station'] == station]
            if len(s_test_df) == 0: continue
            
            s_preds = s_model.predict(s_test_df[features])
            res = evaluate_model(s_test_df[TARGET], s_preds, station)
            plot_results(s_test_df[TARGET], s_preds, station)
            results.append(res)
            
    # 3. Summary Report
    summary_df = pd.DataFrame(results)
    print("\n\n--- Performance Summary Report ---")
    print(summary_df.to_string(index=False))
    
    summary_file = 'performance_test_report.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary report saved to: {summary_file}")

if __name__ == "__main__":
    run_test()
