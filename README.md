# Hyderabad Air Quality Index (AQI) Prediction System



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
To collect live data, you can run the collector in different modes:

- **Single Fetch**: Collect data once and exit.
  ```bash
  python live_data_collector.py --mode once
  ```
- **Daily Automation**: Run in the background and fetch data every 24 hours.
  ```bash
  python live_data_collector.py --mode daily
  ```
- **Hourly Updates**: For high-frequency tracking, fetch every hour.
  ```bash
  python live_data_collector.py --mode hourly
  ```

---

## 🌐 Deployment & 24/7 Operation

This project is designed for continuous monitoring. For professional 24/7 deployment:
- **Process Management**: Use [PM2](https://pm2.keymetrics.io/) to manage the Dashboard and Data Collector processes.
- **Auto-Restart**: PM2 will automatically restart services if they crash or the server reboots.
- **Logging**: Use PM2's built-in logging system (`pm2 logs`) to monitor health.
- **Cloud Hosting**: Recommended for VPS (vultr, AWS, DigitalOcean) for full control over background collectors.

*For detailed instructions, see the [Deployment & Maintenance Guide](.gemini/antigravity/brain/ba8cd5f6-c84e-4cc3-89b5-97fac084c8d3/deployment_guide.md).*

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
