# Hyderabad Air Quality Index (AQI) Prediction System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-232F3E?style=for-the-badge)
![Data Science](https://img.shields.io/badge/Data_Science-Analytics-blue?style=for-the-badge)

An AI-powered environmental forecasting system designed to monitor and predict Air Quality Index (AQI) across various monitoring stations in Hyderabad, India. The project integrates real-time data from the WAQI API with high-performance XGBoost models to provide accurate 24-hour forecasts.

---

## 🌟 Key Features

- **Live Monitoring**: Real-time AQI tracking across multiple Hyderabad stations (Balanagar, HITEC City, Sanathnagar, etc.).
- **AI Forecasting**: Predicts next-day AQI using station-specific XGBoost regression models.
- **Interactive Dashboard**: Premium Streamlit-based UI with:
    - Dynamic entrance animations.
    - City-wide average summary and trend indicators.
    - Interactive station map using Folium.
    - Confidence scores for every prediction.
- **Automated Data Collection**: Continuous background fetching of live data to improve model accuracy over time.
- **Offline Reliability**: Fallback mechanisms to global models if station-specific data is unavailable.

---

## 🏗️ Project Structure

```text
AQI DEMO/
├── dashboard.py           # Main Streamlit dashboard application
├── train_model.py         # Script to train Global and Station-wise XGBoost models
├── predict_live.py        # CLI tool for real-time inference
├── live_data_collector.py  # Service to fetch and store WAQI API data
├── fetch_live_data.py     # Utility to interface with WAQI API
├── map_view.py            # Folium map integration for the dashboard
├── data/                  # Datasets (Historical 10y + Live collected)
├── models/                # Saved .pkl models (Global & Station-specific)
├── plots/                 # Performance visualizations (MAE, Feature Importance)
└── requirements.txt       # Project dependencies
```

---

## 🛠️ Technology Stack

- **ML Framework**: XGBoost, Scikit-learn
- **Data Handling**: Pandas, NumPy
- **Dashboard**: Streamlit, Plotly, Folium
- **API**: World Air Quality Index (WAQI)
- **Visualization**: Matplotlib, Seaborn, Altair

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd aqi-prediction-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the root directory and add your WAQI API Token:
```env
WAQI_API_TOKEN=your_token_here
```

---

## 🔧 Usage

### Training Models
To retrain the global and station-specific models using the latest data:
```bash
python train_model.py
```

### Running the Dashboard
Launch the interactive web interface:
```bash
streamlit run dashboard.py
```

### Live Data Collection
To start collecting live data in the background (helpful for long-term model improvement):
```bash
python live_data_collector.py
```

---

## 📊 Model Performance
The global model currently achieves:
- **Mean Absolute Error (MAE)**: ~6.16
- **R² Score**: ~0.97
*Individual station models often perform even better with lower MAE.*

---

## 📝 License
This project is for educational and demonstration purposes.

---
*Developed with focus on Environmental Intelligence for the city of Hyderabad.*
