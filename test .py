# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt


# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("aqi_data.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ==============================
# 3. Convert Date Column
# ==============================
df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek


# ==============================
# 4. Create Lag Features
# ==============================
df['AQI_lag1'] = df['AQI'].shift(1)
df['AQI_lag3'] = df['AQI'].shift(3)
df['AQI_lag7'] = df['AQI'].shift(7)

df = df.dropna()


# ==============================
# 5. Select Features
# ==============================
features = [
    'Year','Month','Day','DayOfWeek',
    'PM2.5','PM10','NO2','SO2','O3','CO',
    'Temperature','Humidity','Wind_Speed',
    'AQI_lag1','AQI_lag3','AQI_lag7'
]

target = 'AQI'

X = df[features]
y = df[target]


print("\nFeatures used for training:")
print(features)

print("\nTarget Variable:")
print(target)


# ==============================
# 6. Train Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)


# ==============================
# 7. Define Models
# ==============================
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200),
    "XGBoost": XGBRegressor(n_estimators=300),
    "LightGBM": LGBMRegressor(n_estimators=300)
}


# ==============================
# 8. Train & Evaluate Models
# ==============================
results = []

for name, model in models.items():

    print(f"\nTraining {name} model...")
    
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2
    })

    print(f"{name} Performance")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)


# ==============================
# 9. Compare Models
# ==============================
results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df)


# ==============================
# 10. Feature Importance (Best Model)
# ==============================
best_model = models["XGBoost"]

importance = best_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)


# ==============================
# 11. Plot Feature Importance
# ==============================
plt.figure(figsize=(10,6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.title("Feature Importance - XGBoost")
plt.gca().invert_yaxis()
plt.show()