"""
best_aqi_models.py
==================
A professional machine learning pipeline for Air Quality Index (AQI) prediction.
This script performs data loading, feature engineering, multi-model training, 
and detailed performance evaluation.

Author: Senior Machine Learning Engineer
Date: 2026-03-09
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ── 1. Configuration & Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# File path (adjust if necessary)
DATA_PATH = "data/raw/hyderabad_air_quality_10y_combined_fixed.csv"

def main():
    logger.info("═" * 60)
    logger.info("  AQI PREDICTION PIPELINE START")
    logger.info("═" * 60)

    # ── 2. Load Dataset ───────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found at {DATA_PATH}. Please check the path.")
        return

    logger.info(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # ── 3. Dataset Exploration ────────────────────────────────────────────────
    print("\n--- [DATASET INFORMATION] ---")
    print(f"Shape            : {df.shape}")
    print(f"Columns          : {df.columns.tolist()}")
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("-" * 30)

    # ── 4. Feature Engineering ───────────────────────────────────────────────
    logger.info("Starting feature engineering...")
    
    # Convert Date to datetime object
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Station', 'Date'])

    # Create Lag Features (Shifted within each station to avoid leakage)
    logger.info("Creating lag features: AQI_lag1, AQI_lag3, AQI_lag7")
    df['AQI_lag1'] = df.groupby('Station')['AQI'].shift(1)
    df['AQI_lag3'] = df.groupby('Station')['AQI'].shift(3)
    df['AQI_lag7'] = df.groupby('Station')['AQI'].shift(7)

    # Target Variable: Predict next day's AQI
    # (Note: In your previous logic, you might want to predict the CURRENT AQI using lags 
    # OR predict the NEXT DAY. To stay consistent with your requirement, we'll use 
    # Current AQI as target and use previous days as features.)
    
    # Drop rows with NaN values created by lagging
    before_drop = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Dropped {before_drop - len(df)} rows containing NaN after lagging.")

    # Select Features and Target
    features = [
        'Year', 'Month', 'Day', 'DayOfWeek',
        'PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO',
        'Temperature', 'Humidity', 'Wind_Speed', 'Rainfall',
        'AQI_lag1', 'AQI_lag3', 'AQI_lag7'
    ]
    target = 'AQI'

    X = df[features]
    y = df[target]

    print("\n--- [MODEL SPECS] ---")
    print(f"Features  ({len(features)}): {features}")
    print(f"Target Variable : {target}")
    print("-" * 30)

    # ── 5. Train / Test Split ─────────────────────────────────────────────────
    # Since this is time-series data, a chronological split is preferred to avoid 
    # future-lookahead bias.
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    logger.info(f"Training Data Size : {len(X_train)}")
    logger.info(f"Testing Data Size  : {len(X_test)}")

    # ── 6. Model Training & Evaluation ────────────────────────────────────────
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=-1)
    }

    results = []

    for name, model in models.items():
        logger.info(f"Training {name} model...")
        model.fit(X_train, y_train)

        logger.info(f"Evaluating {name} model...")
        preds = model.predict(X_test)

        # Performance Metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            "Model": name,
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "R2_Score": round(r2, 4)
        })

    # ── 7. Model Comparison Table ─────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    print("\n--- [MODEL PERFORMANCE COMPARISON] ---")
    print(results_df.to_string(index=False))
    print("-" * 30)

    # ── 8. Feature Importance (Best Model) ────────────────────────────────────
    best_model_name = results_df.iloc[0]["Model"]
    logger.info(f"Best Model based on RMSE: {best_model_name}")
    
    best_model = models[best_model_name]
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        print("\n--- [FEATURE IMPORTANCE - BEST MODEL] ---")
        print(importance_df.head(10).to_string(index=False))
        print("-" * 30)

        # ── 9. Plot Feature Importance ────────────────────────────────────────
        plt.figure(figsize=(12, 7))
        sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
        plt.title(f"Feature Importance - {best_model_name}", fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        # Save plot if needed
        # plt.savefig("feature_importance.png")
        logger.info("Displaying feature importance plot...")
        plt.show()

    logger.info("═" * 60)
    logger.info("  AQI PREDICTION PIPELINE COMPLETE")
    logger.info("═" * 60)

if __name__ == "__main__":
    main()
