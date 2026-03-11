import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# 1. Configuration
DATA_PATH = 'data/hyderabad_air_quality_10y_combined_fixed.csv'
MODEL_PATH = 'models/trained_model.pkl'
STATION_MODELS_DIR = 'models/station_models'
PLOTS_DIR = 'plots'
FEATURES = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO', 'Temperature', 'Humidity', 'Wind_Speed', 'Rainfall', 'Station_Code', 'AQI_Lag_1', 'AQI_Lag_2', 'AQI_Rolling_3']
TARGET = 'AQI'

def engineer_features(df):
    """Create lag features."""
    df = df.sort_values(['Station', 'Date'])
    
    # Create Lag Features (Yesterday and Day Before Yesterday)
    df['AQI_Lag_1'] = df.groupby('Station')['AQI'].shift(1)
    df['AQI_Lag_2'] = df.groupby('Station')['AQI'].shift(2)
    
    # Create Rolling Average (Last 3 days including today's context)
    # Note: Using shift(1) for rolling mean to simulate predicting 'tomorrow' using only values up to 'today'
    df['AQI_Rolling_3'] = df.groupby('Station')['AQI'].transform(lambda x: x.shift(1).rolling(window=3).mean())
    
    # Drop rows with NaN values created by shifts
    df = df.dropna(subset=['AQI_Lag_1', 'AQI_Lag_2', 'AQI_Rolling_3'])
    
    return df

def plot_performance(y_test, preds, model, features, station_name='Global'):
    """Generate Actual vs Predicted and Feature Importance plots."""
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    
    suffix = station_name.lower().replace(' ', '_')
    
    # 1. Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, preds, alpha=0.5, color='royalblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title(f'Actual vs Predicted AQI ({station_name} - XGBoost)')
    plt.savefig(os.path.join(PLOTS_DIR, f'actual_vs_predicted_{suffix}.png'), dpi=150)
    plt.close()
    
    # 2. Feature Importance
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title(f'Feature Importances ({station_name} - XGBoost)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'feature_importance_{suffix}.png'), dpi=150)
    plt.close()

def train():
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists(STATION_MODELS_DIR):
        os.makedirs(STATION_MODELS_DIR)

    # Load data
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        # Check if it's still in the old location if move failed
        if os.path.exists('hyderabad_air_quality_10y_combined_fixed.csv'):
            if not os.path.exists('data'): os.makedirs('data')
            os.rename('hyderabad_air_quality_10y_combined_fixed.csv', DATA_PATH)
        else:
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    
    # Engineer features part 1: Station Code
    df['Station_Code'] = df['Station'].astype('category').cat.codes
    
    # Clean data (ensure pollutants and target are not null)
    core_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO', 'Temperature', 'Humidity', 'Wind_Speed', 'Rainfall']
    df = df.dropna(subset=[TARGET] + core_features) # Drop if target or core pollutants are missing
    
    # Engineer features part 2: Lags and Rolling
    df = engineer_features(df)
    
    # --- 1. Train Global Model ---
    print("\n--- Training Global Model ---")
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Global Model Performance: MAE={mae:.2f}, R2={r2:.4f}")
    plot_performance(y_test, preds, model, FEATURES, 'Global')
    
    station_mapping = {i: cat for i, cat in enumerate(df['Station'].astype('category').cat.categories)}
    payload = {
        'model': model,
        'features': FEATURES,
        'station_mapping': station_mapping
    }
    joblib.dump(payload, MODEL_PATH)
    print("Global model saved.")

    # --- 2. Train Station-wise Models ---
    print("\n--- Training Station-wise Models ---")
    stations = df['Station'].unique()
    for station in stations:
        print(f"Training model for station: {station}")
        station_df = df[df['Station'] == station]
        
        if len(station_df) < 50: # Skip if too little data
            print(f"  Skipping {station} (not enough data: {len(station_df)})")
            continue
            
        X_s = station_df[FEATURES]
        y_s = station_df[TARGET]
        
        # Split station data
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y_s, test_size=0.25, random_state=42, shuffle=False)
        
        # Train
        model_s = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        model_s.fit(X_train_s, y_train_s)
        
        # Evaluate
        preds_s = model_s.predict(X_test_s)
        mae_s = mean_absolute_error(y_test_s, preds_s)
        print(f"  Performance: MAE={mae_s:.2f}")
        
        # Save Station Model
        station_file = os.path.join(STATION_MODELS_DIR, f"{station.lower().replace(' ', '_')}.pkl")
        station_payload = {
            'model': model_s,
            'features': FEATURES,
            'station_name': station
        }
        joblib.dump(station_payload, station_file)

    print("\nModel training successful and saved.")

if __name__ == "__main__":
    train()

if __name__ == "__main__":
    train()
