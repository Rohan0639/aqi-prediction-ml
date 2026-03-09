# AQI Prediction System

A **production-style Machine Learning system** that predicts the next day's Air Quality Index (AQI) for Hyderabad monitoring stations.

---

## 📁 Project Structure

```
AQI DEMO/
├── config/
│   └── config.yaml          ← All settings (paths, hyperparameters, API key)
├── data/
│   ├── raw/                 ← Original CSV dataset
│   └── processed/           ← Cleaned dataset + output plots (PNG)
├── models/
│   └── trained_models/      ← Saved Joblib models + metadata JSON
├── notebooks/
│   └── EDA.ipynb            ← Exploratory Data Analysis
├── pipeline/
│   └── run_pipeline.py      ← ⭐ One-command training pipeline
├── src/
│   ├── data_loader.py       ← Load & validate CSV
│   ├── preprocessing.py     ← Clean, impute, clip outliers
│   ├── feature_engineering.py ← Lag, rolling, seasonal, interaction features
│   ├── train_model.py       ← Linear Regression + XGBoost trainer
│   ├── evaluate_model.py    ← RMSE / MAE / R² + 3 plots
│   ├── fetch_api_data.py    ← Real-time WAQI API data fetcher
│   └── predict.py           ← Live next-day AQI predictor
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train all models (one command)

```bash
python pipeline/run_pipeline.py
```

This will:
- Load and clean 10 years of AQI data
- Create lag / rolling / seasonal features
- Train **Linear Regression** and **XGBoost**
- Evaluate with RMSE, MAE, R²
- Save models to `models/trained_models/`
- Generate 3 diagnostic plots in `data/processed/`

### 3. Predict tomorrow's AQI

```bash
# Using live API data
python src/predict.py

# Offline fallback (uses last row of processed data)
python src/predict.py --offline
```

---

## 🛠️ Module Descriptions

| File | Purpose |
|------|---------|
| `data_loader.py` | Reads raw CSV, parses dates, validates shape |
| `preprocessing.py` | Drops duplicates, forward-fills missing values, IQR outlier clipping |
| `feature_engineering.py` | Creates lag(1,2,3,7) + rolling mean/std(3,7,14) + cyclical time + pollutant ratios + **Next_day_AQI** target |
| `train_model.py` | Chronological 80/20 split, fits LR (with StandardScaler) & XGBoost, saves models |
| `evaluate_model.py` | RMSE/MAE/R² table, selects best model, saves 3 PNG plots |
| `fetch_api_data.py` | Queries WAQI API stations, converts JSON → training-compatible row |
| `predict.py` | Loads best model, builds feature vector from live + history, prints forecast |
| `run_pipeline.py` | Orchestrates steps 1–6 end-to-end with logging and timing |

---

## 📊 Features Created

| Category | Features |
|----------|---------|
| **Lag** | `AQI_lag1`, `AQI_lag2`, `AQI_lag3`, `AQI_lag7` |
| **Rolling** | `AQI_rolling_mean_{3,7,14}`, `AQI_rolling_std_{3,7,14}` |
| **Pollutants** | `PM2.5`, `PM10`, `NO2`, `SO2`, `O3`, `CO` |
| **Weather** | `Temperature`, `Humidity`, `Wind_Speed`, `Rainfall` |
| **Seasonal** | `Month_sin/cos`, `DOW_sin/cos`, `Season`, `Is_Weekend`, `DayOfYear`, `Week` |
| **Interactions** | `PM_ratio`, `NOx_SO2`, `Heat_Humidity`, `Wind_Rain`, `CO_NO2_ratio` |
| **Station** | `Station_code` (label encoded) |

---

## 🔑 API Configuration

The WAQI API key is pre-configured in `config/config.yaml`:
```yaml
api:
  waqi_token: "73bef76263763e96bb7672aa0ba6c5b6fca973e7"
  default_city: "hyderabad"
```

---

## 📈 Example Output

```
══════════════════════════════════════════════════
   🌡  AQI PREDICTION REPORT
══════════════════════════════════════════════════
  Station           : Hyderabad
  Current AQI       : 142.0  (Moderate)
  ─────────────────────────────────────────
  XGBoost Prediction: 138.5  (Moderate)
  Linear Reg. Pred. : 145.2  (Moderate)
  ─────────────────────────────────────────
  Best Model        : XGBoost
  Predicted AQI Tomorrow : 138.5  (Moderate)
══════════════════════════════════════════════════
```

---

## 🔬 Model Comparison

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | ~45 | ~32 | ~0.76 |
| **XGBoost** | **~28** | **~19** | **~0.91** |

*(Results vary by station and date range)*

---

## 💡 Suggestions for Improving Accuracy

1. **Station-specific models** – Train one XGBoost per station instead of pooling all stations.
2. **More lags** – Add AQI_lag14, AQI_lag30 for trend memory.
3. **External weather API** – Fetch tomorrow's weather forecast to add as features.
4. **LightGBM / CatBoost** – Often outperform XGBoost on tabular time-series tasks.
5. **LSTM / Transformer** – Sequence models can capture longer temporal dependencies.
6. **Hyperparameter tuning** – Use Optuna or GridSearchCV within the training set.
7. **Ensemble** – Blend XGBoost + LR predictions using a meta-learner.

---

## 📜 License

Open-source for educational and research purposes.
