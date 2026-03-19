import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from fetch_live_data import get_live_data, STATION_MAP
from map_view import render_map_view

# ---------------------------------------------------------
# UI CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hyderabad AQI Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Auto-refresh to 2 minutes (120,000 milliseconds)
count = st_autorefresh(interval=120000, limit=None, key="dashboard_autorefresh")

# ---------------------------------------------------------
# REBUILT INTRO ANIMATION SYSTEM
# ---------------------------------------------------------
if 'intro_played' not in st.session_state:
    st.session_state.intro_played = False

def render_intro_system():
    title_text = "Air Quality Prediction System"
    # Create staggered spans for the title layout via python instead of JS
    spans = "".join([f"<span style='animation-delay: {0.3 + i*0.03}s'>{char if char != ' ' else '&nbsp;'}</span>" for i, char in enumerate(title_text)])

    intro_html = f"""
    <!-- Pure CSS Intro Screen -->
    <div id="intro-screen">
        <div class="intro-content">
            <h1 class="stagger-title">
                {spans}
            </h1>
            <p id="animated-subtitle">Designed for Hyderabad</p>
            <div class="loader-container">
                <div class="loader-bar"></div>
            </div>
        </div>
    </div>

    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* 1. Intro Screen Overlay - Fluid Dark Premium Background */
    #intro-screen {{
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: linear-gradient(-45deg, #020617, #0f172a, #082f49, #020617);
        background-size: 400% 400%;
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999999;
        font-family: 'Outfit', sans-serif;
        
        /* Subtle background movement */
        animation: gradientBG 8s ease infinite, hideIntro 1s cubic-bezier(0.65, 0, 0.35, 1) forwards;
        animation-delay: 0s, 3.5s; 
    }}

    @keyframes gradientBG {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    @keyframes hideIntro {{
        to {{
            opacity: 0;
            visibility: hidden;
            pointer-events: none;
            display: none;
            transform: scale(1.05); /* Slight push-in effect on exit */
        }}
    }}

    .intro-content {{
        text-align: center;
        z-index: 10;
        padding: 40px;
        width: 100%;
        max-width: 1200px;
    }}

    /* 2. Premium Title Animation: Blur + Scale + Fade */
    .stagger-title {{
        font-size: 4.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }}

    .stagger-title span {{
        display: inline-block;
        opacity: 0;
        transform: translateY(15px) scale(0.95);
        filter: blur(8px);
        animation: premiumLetterIn 1s cubic-bezier(0.22, 1, 0.36, 1) forwards;
    }}

    @keyframes premiumLetterIn {{
        to {{
            opacity: 1;
            transform: translateY(0) scale(1);
            filter: blur(0);
        }}
    }}

    /* 3. Sleek Subtitle Animation */
    #animated-subtitle {{
        font-size: 1.4rem;
        color: #60a5fa; /* Soft blue */
        font-weight: 300;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0;
        transform: translateY(10px);
        animation: subtitleIn 1.5s cubic-bezier(0.22, 1, 0.36, 1) forwards;
        animation-delay: 1.8s;
    }}

    @keyframes subtitleIn {{
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    /* 4. Minimalist Loader */
    .loader-container {{
        width: 240px;
        height: 2px;
        background: rgba(255, 255, 255, 0.1);
        margin: 3rem auto 0;
        border-radius: 2px;
        overflow: hidden;
        opacity: 0;
        animation: fadeIn 1s ease forwards;
        animation-delay: 1.5s;
    }}

    .loader-bar {{
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #10b981);
        transform: translateX(-100%);
        animation: loadProgress 2.5s cubic-bezier(0.65, 0, 0.35, 1) forwards;
        animation-delay: 1.5s;
    }}

    @keyframes fadeIn {{
        to {{ opacity: 1; }}
    }}

    @keyframes loadProgress {{
        to {{ transform: translateX(0); }}
    }}

    /* Mobile fixes */
    @media (max-width: 768px) {{
        .stagger-title {{ font-size: 2.5rem; }}
        #animated-subtitle {{ font-size: 1rem; }}
        .loader-container {{ width: 160px; }}
    }}
    </style>
    """
    st.markdown(intro_html, unsafe_allow_html=True)
# Apply Intro only once per session
if not st.session_state.intro_played:
    st.session_state.just_played_intro = True
    render_intro_system()
    st.session_state.intro_played = True
else:
    st.session_state.just_played_intro = False

st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 3rem !important;
        font-weight: 900 !important;
    }
    .main-metric-container {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #111827; 
        border: 1px solid #1f2937;
        margin-bottom: 1rem;
    }
    .metric-title {
        color: #9ca3af;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 4rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .metric-category {
        font-size: 1.25rem;
        font-weight: 700;
    }
    
    /* Live Status Badge Styling */
    .live-badge-container {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 4px 12px;
        border-radius: 9999px;
        margin-bottom: 1rem;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .live-text {
        color: #10b981;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .update-time {
        color: #94a3b8;
        font-size: 0.75rem;
        margin-left: 8px;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# BACKEND LOGIC INTEGRATION
# ---------------------------------------------------------
MODEL_PATH = 'models/trained_model.pkl'
STATION_MODELS_DIR = 'models/station_models'

def get_aqi_category_color(aqi):
    if aqi is None: return "gray"
    if aqi <= 50: return "#22c55e" # Green
    if aqi <= 100: return "#eab308" # Yellow
    if aqi <= 150: return "#f97316" # Orange
    if aqi <= 200: return "#ef4444" # Red
    if aqi <= 300: return "#a855f7" # Purple
    return "#9f1239" # Maroon

def clean_station_name(name):
    """Remove SPCB, commas, and trailing spaces from station names."""
    return name.replace(" SPCB", "").replace(",", "").strip()

def get_aqi_category(aqi):
    if aqi is None: return "Unknown"
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

@st.cache_resource
def load_all_models():
    """Load the global model once, to get features and mappings."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except:
             return None
    return None

def load_station_model(station_name):
    """Load station-specific model or fallback to global."""
    station_file = os.path.join(STATION_MODELS_DIR, f"{station_name.lower().replace(' ', '_')}.pkl")
    if os.path.exists(station_file):
        try:
            return joblib.load(station_file)
        except:
             pass
    
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except:
             pass
    return None

def calculate_dominant_pollutant(live_record):
    """A simple heuristic to find dominant pollutant from live readings"""
    pollutants = {k: live_record.get(k, 0) for k in ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']}
    if not pollutants: return "Unknown"
    return max(pollutants.keys(), key=lambda k: pollutants[k])

# ---------------------------------------------------------
# WEATHER CSV LOADER
# ---------------------------------------------------------

WEATHER_CSV_PATH = 'data/hyderabad_live_weather.csv'

WIND_ICON_MAP = {
    'N':  '⬆️',
    'NE': '↗️',
    'E':  '➡️',
    'SE': '↘️',
    'S':  '⬇️',
    'SW': '↙️',
    'W':  '⬅️',
    'NW': '↖️',
}

@st.cache_data(ttl=120)
def load_weather_from_csv():
    """
    Load the latest weather row per station from the CSV file.
    Returns a dict keyed by normalised station name → weather dict.
    TTL=120 keeps it in sync with the dashboard auto-refresh.
    """
    if not os.path.exists(WEATHER_CSV_PATH):
        return {}
    try:
        df = pd.read_csv(WEATHER_CSV_PATH)
        # Combine Date + Time into a single sortable datetime
        df['_dt'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                                   errors='coerce')
        # Keep only the latest record per station
        df = df.sort_values('_dt').groupby('Station', as_index=False).last()

        weather_map = {}
        for _, row in df.iterrows():
            raw_name = str(row.get('Station', '')).strip()
            # Normalise: lowercase, remove 'spcb', collapse spaces
            norm_name = raw_name.lower().replace('spcb', '').replace(',', '').strip()
            weather_map[norm_name] = {
                'raw_station':      raw_name,
                'temperature':      row.get('Temperature', '--'),
                'humidity':         row.get('Humidity',    '--'),
                'pressure':         row.get('Pressure',    '--'),
                'wind_speed':       row.get('Wind_Speed',  '--'),
                'wind_dir_deg':     row.get('Wind_Direction_Deg',   '--'),
                'wind_dir_label':   str(row.get('Wind_Direction_Label', '--')).strip().upper(),
                'rainfall':         row.get('Rainfall',    '--'),
            }
        return weather_map
    except Exception as e:
        print(f"[WeatherCSV] Error loading: {e}")
        return {}


def get_csv_weather_for_station(station_name, weather_map):
    """
    Fuzzy-match `station_name` against the normalised keys in `weather_map`.
    Returns the matched weather dict, or an empty dict if no match found.
    """
    # Normalise query the same way as the loader
    query = station_name.lower().replace('spcb', '').replace(',', '').strip()

    # Exact normalised match
    if query in weather_map:
        return weather_map[query]

    # Partial / substring match (handles minor spacing differences)
    for key, val in weather_map.items():
        if query in key or key in query:
            return val

    return {}


@st.cache_data(ttl=120)  # Cache for 2 mins to match auto-refresh
def get_dashboard_data():
    """
    Connects to the backend functions (predict_live, fetch_live_data)
    and formats them for the Streamlit dashboard UI.
    """
    global_payload = load_all_models()
    if not global_payload:
        return {}, [], []

    features = global_payload.get('features', [])
    
    # Extract station mapping and reverse it (from name to code)
    station_mapping = global_payload.get('station_mapping', {})
    name_to_code = {name: code for code, name in station_mapping.items()}
    
    # 1. Fetch Current Data for all mapped stations
    stations_current = []
    stations_prediction = []
    all_station_details = {}
    
    # Use only live_aqi_dataset.csv for recent lag data (T-1, T-2)
    live_file = 'data/live_aqi_dataset.csv'

    try:
        if os.path.exists(live_file):
            df_all = pd.read_csv(live_file, parse_dates=['Date'])
            df_all = df_all.drop_duplicates(subset=['Date', 'Station'], keep='last')
        else:
            df_all = pd.DataFrame()
    except:
        df_all = pd.DataFrame()

    
    for station_name in STATION_MAP.keys():
        try:
            # -- FETCH LIVE --
            live_data = get_live_data(station_name)
            if not live_data:
                continue
                
            # Current AQI for display (using the original sub-index/IAQI)
            current_aqi_val = int(live_data.get('AQI', 0)) 
            dominant = calculate_dominant_pollutant(live_data)
            
            stations_current.append({
                'Station': station_name,
                'Current AQI': current_aqi_val,
                'Category': get_aqi_category(current_aqi_val),
                'Dominant Pollutant': dominant
            })

            # -- RUN PREDICTION FOR TOMORROW --
            payload = load_station_model(station_name)
            if payload and 'model' in payload:
                model = payload['model']
                
                # -- PREPARE LAGS: T0 (Today) and T-1 (Yesterday) --
                today_aqi = current_aqi_val
                aqi_yest = today_aqi # Fallback
                aqi_db_yest = today_aqi # Fallback
                
                if not df_all.empty:
                    # Filter for station and ensure we only use records BEFORE today's system date for history
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    df_station = df_all[
                        (df_all['Station'] == station_name) & 
                        (df_all['Date'].dt.strftime('%Y-%m-%d') < today_str)
                    ].sort_values('Date')
                    
                    if len(df_station) >= 2:
                        hist_vals = df_station.tail(2)['AQI'].tolist()
                        aqi_yest = hist_vals[1] # T-1
                        aqi_db_yest = hist_vals[0] # T-2

                # Calculate Rolling 3: (Today + Yesterday + DayBefore) / 3
                rolling_3 = (today_aqi + aqi_yest + aqi_db_yest) / 3

                # Prepare DataFrame for prediction matching exact model features
                live_data['AQI_Lag_1'] = today_aqi  # Lag 1 is NOW Today
                live_data['AQI_Lag_2'] = aqi_yest   # Lag 2 is NOW Yesterday
                live_data['AQI_Rolling_3'] = rolling_3
                
                # Fetch dynamically correct Station_Code mapping from global model payload
                live_data['Station_Code'] = name_to_code.get(station_name, 1)
                
                # Ensure correct feature order
                # PM2.5 and PM10 are already concentrations in live_data (updated in fetch_live_data.py)
                input_dict = {f: live_data.get(f, 0) for f in features}
                df_input = pd.DataFrame([input_dict])

                # Predict
                prediction = float(model.predict(df_input)[0])
                
                stations_prediction.append({
                    'Station': station_name,
                    'Predicted AQI': int(prediction),
                    'Category': get_aqi_category(int(prediction)),
                    'Model Used': 'Station-specific' if 'station_name' in payload else 'Global Fallback'
                })
                
                # Use REAL accuracy from logs
                # Global R2 is 0.9791 -> ~97.9%
                # We'll use 97.9% for global fallback, and calculate a weight for station models
                if 'station_name' in payload:
                    # Station models in logs show MAE ~3-5. Smaller MAE = higher confidence.
                    # We'll map MAE to a confidence percentage (Heuristic: 100 - MAE*1.2)
                    mae_map = {
                        'Balanagar SPCB': 4.89, 'HITEC City': 3.70, 
                        'IDA Pashamylaram SPCB': 5.72, 'Sanathnagar SPCB': 4.80, 
                        'US Consulate': 3.07, 'Uppal SPCB': 4.13, 'Zoo Park SPCB': 4.22
                    }
                    base_mae = mae_map.get(station_name, 4.5)
                    confidence_val = round(100 - (base_mae * 1.5), 1)
                else:
                    confidence_val = 97.9
                
                all_station_details[station_name] = {
                    'current_aqi': current_aqi_val,
                    'dominant_pollutant': dominant,
                    'predicted_aqi': int(prediction),
                    'model_used': 'Station-specific' if 'station_name' in payload else 'Global Fallback',
                    'confidence_score': confidence_val,
                    'mae': mae_map.get(station_name, 6.16) if 'station_name' in payload else 6.16,
                    'weather': {
                        'temperature': live_data.get('Temperature', '--') if isinstance(live_data, dict) else '--',
                        'humidity': live_data.get('Humidity', '--') if isinstance(live_data, dict) else '--',
                        'rain': live_data.get('Rainfall', '--') if isinstance(live_data, dict) else '--'
                    }
                }

        except Exception as e:
            print(f"Skipping {station_name} due to error: {e}")
            continue

    # Return all station details
    return all_station_details, stations_current, stations_prediction


# ---------------------------------------------------------
# RENDER UI
# ---------------------------------------------------------

st.divider()

st.markdown(f"""
<div class="dashboard-header">
    <h1 class="main-title">Hyderabad AQI Dashboard</h1>
    <p class="sub-title">Real-Time Forecasts & Insights</p>
    <div class="live-badge-container">
        <div class="pulse-dot"></div>
        <span class="live-text">Live System Status</span>
        <span class="update-time">Last Updated: {datetime.now().strftime('%H:%M:%S')}</span>
    </div>
</div>
<style>
.dashboard-header {{
    text-align: center;
    padding: 1rem 0 2rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}}
.main-title {{
    font-size: 3rem;
    font-weight: 900;
    color: #f8fafc;
    margin-bottom: 0.5rem;
}}
.sub-title {{
    font-size: 1.1rem;
    color: #3b82f6;
}}
@media (max-width: 768px) {{
    .main-title {{ font-size: 2.2rem; }}
    .sub-title {{ font-size: 1rem; }}
}}
</style>
""", unsafe_allow_html=True)


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Hyderabad Station Map"])

with st.spinner("Executing Models and Fetching WAQI Live Environmental Data..."):
    all_station_details, stations_current, stations_prediction = get_dashboard_data()

if page == "Hyderabad Station Map":
    render_map_view(all_station_details)
    st.stop()

if all_station_details:
    # --- CALCULATE CITY AVERAGE AQI ---
    valid_current_aqi = [data['current_aqi'] for data in all_station_details.values() if isinstance(data['current_aqi'], (int, float))]
    valid_pred_aqi = [data['predicted_aqi'] for data in all_station_details.values() if isinstance(data['predicted_aqi'], (int, float))]
    
    avg_current_aqi = int(sum(valid_current_aqi) / len(valid_current_aqi)) if valid_current_aqi else None
    avg_pred_aqi = int(sum(valid_pred_aqi) / len(valid_pred_aqi)) if valid_pred_aqi else None

    # --- TOP AVERAGE SUMMARY CARD ---
    st.markdown("### 🌆 Hyderabad City Summary")
    
    avg_col1, avg_col2 = st.columns(2, gap="medium")
    
    with avg_col1:
        cat_curr = get_aqi_category(avg_current_aqi) if avg_current_aqi else '--'
        color_curr = get_aqi_category_color(avg_current_aqi) if avg_current_aqi else 'gray'
        st.markdown(f"""
        <div class="city-avg-card" style="border-top: 4px solid {color_curr};">
            <div class="avg-label">Average Live AQI</div>
            <div class="avg-value" style="color: {color_curr};">{avg_current_aqi if avg_current_aqi else '--'}</div>
            <div class="avg-cat" style="background-color: {color_curr}20; color: {color_curr};">{cat_curr}</div>
        </div>
        """, unsafe_allow_html=True)

    with avg_col2:
        cat_pred = get_aqi_category(avg_pred_aqi) if avg_pred_aqi else '--'
        color_pred = get_aqi_category_color(avg_pred_aqi) if avg_pred_aqi else 'gray'
        
        trend_arrow = ""
        trend_text = ""
        if avg_current_aqi and avg_pred_aqi:
            if avg_pred_aqi > avg_current_aqi:
                trend_arrow = "↑"
                trend_text = "Expected to worsens"
                t_color = "#ef4444"
            elif avg_pred_aqi < avg_current_aqi:
                trend_arrow = "↓"
                trend_text = "Improvement expected"
                t_color = "#22c55e"
            else:
                trend_arrow = "→"
                trend_text = "Staying stable"
                t_color = "#9ca3af"

        st.markdown(f"""
        <div class="city-avg-card" style="border-top: 4px solid {color_pred};">
            <div class="avg-label">Forecasted Average (24h)</div>
            <div class="avg-value" style="color: {color_pred};">
                {avg_pred_aqi if avg_pred_aqi else '--'}
                <span style="font-size: 1.5rem; vertical-align: middle; margin-left: 5px; color: {t_color if trend_text else color_pred};">{trend_arrow}</span>
            </div>
            <div class="avg-cat" style="color: {t_color if trend_text else color_pred}; font-size: 0.9rem;">{trend_text if trend_text else cat_pred}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Inject styling for the city-avg-card
    st.markdown("""
    <style>
    .city-avg-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .city-avg-card:hover {
        transform: translateY(-5px);
        background: rgba(30, 41, 59, 0.7);
    }
    .avg-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .avg-value {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .avg-cat {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.divider()

    # --- STATION SELECTION ---
    st.markdown('<h3 style="color: #60a5fa; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">SELECT MONITORING SYSTEM</h3>', unsafe_allow_html=True)
    
    # Create a mapping of clean names to original keys for selection
    station_names = list(all_station_details.keys())
    clean_to_orig = {clean_station_name(name): name for name in station_names}
    
    selected_clean_name = st.selectbox(
        "Choose a station to view its specific data", 
        options=list(clean_to_orig.keys()), 
        label_visibility="collapsed"
    )
    
    selected_station = clean_to_orig[selected_clean_name]
    station_data = all_station_details[selected_station]
    clean_display_name = selected_clean_name

    # --- TOP SECTIONS ---
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader(f"Current AQI ({clean_display_name})")
        aqi_val = station_data.get('current_aqi', '--')
        category = get_aqi_category(aqi_val) if isinstance(aqi_val, (int, float)) else '--'
        pollutant = station_data.get('dominant_pollutant', '--')
        color = get_aqi_category_color(aqi_val if isinstance(aqi_val, (int, float)) else None)
        
        st.markdown(f"""
        <div class="main-metric-container" style="border-left: 6px solid {color};">
            <div class="metric-title">Live API Reading</div>
            <div class="metric-value" style="color: {color};">{aqi_val}</div>
            <div class="metric-category" style="color: {color};">Category: {category}</div>
            <div style="color: #d1d5db; margin-top: 0.5rem; font-weight: 500;">Dominant Pollutant: {pollutant}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Tomorrow AQI Prediction")
        pred_aqi = station_data.get('predicted_aqi', '--')
        pred_cat = get_aqi_category(pred_aqi) if isinstance(pred_aqi, (int, float)) else '--'
        model_name = station_data.get('model_used', '--')
        confidence = station_data.get('confidence_score', '--')
        color = get_aqi_category_color(pred_aqi if isinstance(pred_aqi, (int, float)) else None)
        
        # Determine confidence color
        conf_color = "#22c55e" if isinstance(confidence, float) and confidence >= 90 else "#eab308" if isinstance(confidence, float) and confidence >= 80 else "#ef4444"
        
        st.markdown(f"""
        <div class="main-metric-container" style="border-left: 6px solid {color};">
            <div class="metric-title">AI Forecast</div>
            <div class="metric-value" style="color: {color};">{pred_aqi}</div>
            <div class="metric-category" style="color: {color};">Category: {pred_cat}</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
                <div style="color: #60a5fa; font-weight: 500;">Model: {model_name}</div>
                <div style="color: {conf_color}; font-weight: 700; background-color: {conf_color}20; padding: 2px 8px; border-radius: 4px;">🎯 Confidence: {confidence}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- WEATHER SECTION (CSV) — dynamic per station ---
    st.markdown("### 🌬️ Live Weather Conditions")

    # Load the full weather map (cached globally, TTL=120 s)
    _wx_map  = load_weather_from_csv()

    # Per-station lookup — re-evaluated every time selected_station changes
    _wx_data = get_csv_weather_for_station(selected_station, _wx_map)

    # Pull values for current station
    _temperature    = _wx_data.get('temperature',    '--')
    _humidity       = _wx_data.get('humidity',       '--')
    _wind_speed     = _wx_data.get('wind_speed',     '--')
    _wind_label     = _wx_data.get('wind_dir_label', '--')
    _wind_deg       = _wx_data.get('wind_dir_deg',   0)
    _wind_icon      = WIND_ICON_MAP.get(str(_wind_label).strip().upper(), '🧭')

    # Safely convert degree to float for CSS rotation
    try:
        _deg_val = float(_wind_deg)
    except (ValueError, TypeError):
        _deg_val = 0.0

    # Format display values
    _temp_display     = f"{_temperature} °C" if _temperature != '--' else '--'
    _humidity_display = f"{_humidity} %"     if _humidity    != '--' else '--'
    _speed_display    = f"{_wind_speed} m/s" if _wind_speed  != '--' else '--'

    # Station data source hint — updates as user switches station
    _src_label = _wx_data.get('raw_station', selected_station)
    st.caption(f"📍 Showing weather for: **{_src_label}**")

    # --- ROW 1: Temperature | Humidity ---
    wx_r1c1, wx_r1c2 = st.columns(2)

    with wx_r1c1:
        st.markdown(f"""
        <div class="wx-card wx-temp">
            <div class="wx-label">🌡️ Temperature</div>
            <div class="wx-value">{_temp_display}</div>
        </div>
        """, unsafe_allow_html=True)

    with wx_r1c2:
        st.markdown(f"""
        <div class="wx-card wx-humid">
            <div class="wx-label">💧 Humidity</div>
            <div class="wx-value">{_humidity_display}</div>
        </div>
        """, unsafe_allow_html=True)

    # Spacing between rows
    st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)

    # --- ROW 2: Wind Speed | Wind Direction (rotating arrow) | empty col ---
    wx_r2c1, wx_r2c2, wx_r2c3 = st.columns([1, 1, 1])

    with wx_r2c1:
        st.markdown(f"""
        <div class="wx-card wx-wind">
            <div class="wx-label">💨 Wind Speed</div>
            <div class="wx-value">{_speed_display}</div>
        </div>
        """, unsafe_allow_html=True)

    with wx_r2c2:
        st.markdown(f"""
        <div class="wx-card wx-wind">
            <div class="wx-label">🧭 Wind Direction</div>
            <div class="wx-direction-row">
                <span class="wx-compass" style="transform: rotate({_deg_val}deg);">⬆️</span>
                <span class="wx-dir-text">
                    <span class="wx-dir-icon">{_wind_icon}</span>
                    <span>{_wind_label}&nbsp;({int(_deg_val)}°)</span>
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with wx_r2c3:
        # Placeholder to maintain grid symmetry
        st.markdown("<div style='min-height:110px;'></div>", unsafe_allow_html=True)

    # Weather card styles
    st.markdown("""
    <style>
    .wx-card {
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        transition: transform 0.25s ease, background 0.25s ease;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 0.5rem;
    }
    .wx-card:hover {
        transform: translateY(-4px);
        background: rgba(30, 41, 59, 0.75);
    }
    /* Accent top borders per card type */
    .wx-temp  { border-top: 2px solid #f97316; }
    .wx-humid { border-top: 2px solid #38bdf8; }
    .wx-wind  { border-top: 2px solid #34d399; }

    .wx-label {
        color: #94a3b8;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-weight: 600;
    }
    .wx-value {
        color: #f1f5f9;
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .wx-direction-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        flex-wrap: wrap;
    }
    .wx-compass {
        font-size: 2.2rem;
        display: inline-block;
        filter: drop-shadow(0 0 6px rgba(52, 211, 153, 0.6));
    }
    .wx-dir-text {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 0.15rem;
    }
    .wx-dir-icon { font-size: 1.5rem; }
    .wx-dir-text > span:last-child {
        color: #34d399;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    </style>
    """, unsafe_allow_html=True)


    # --- AQI STANDARDS TABLE ---
    st.markdown("""
    <div style="margin-top: 2rem;">
        <h3 style="margin-bottom: 1rem;">Air Quality Index (AQI) Categories</h3>
        <table style="width:100%; border-collapse: collapse; background-color: #0f172a; border-radius: 12px; overflow: hidden; font-family: 'Outfit', sans-serif;">
            <thead>
                <tr style="background-color: #1e293b; text-align: left;">
                    <th style="padding: 12px 20px; color: #94a3b8; font-weight: 600;">AQI Range</th>
                    <th style="padding: 12px 20px; color: #94a3b8; font-weight: 600;">Category</th>
                    <th style="padding: 12px 20px; color: #94a3b8; font-weight: 600; text-align: center;">Indicator</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px 20px; color: #f8fafc; font-weight: 500;">0 - 50</td>
                    <td style="padding: 12px 20px; color: #22c55e; font-weight: 700;">Good</td>
                    <td style="padding: 12px 20px; text-align: center;"><div style="width: 14px; height: 14px; background-color: #22c55e; border-radius: 50%; display: inline-block;"></div></td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px 20px; color: #f8fafc; font-weight: 500;">51 - 100</td>
                    <td style="padding: 12px 20px; color: #eab308; font-weight: 700;">Moderate</td>
                    <td style="padding: 12px 20px; text-align: center;"><div style="width: 14px; height: 14px; background-color: #eab308; border-radius: 50%; display: inline-block;"></div></td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px 20px; color: #f8fafc; font-weight: 500;">101 - 150</td>
                    <td style="padding: 12px 20px; color: #f97316; font-weight: 700;">Unhealthy for Sensitive Groups</td>
                    <td style="padding: 12px 20px; text-align: center;"><div style="width: 14px; height: 14px; background-color: #f97316; border-radius: 50%; display: inline-block;"></div></td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px 20px; color: #f8fafc; font-weight: 500;">151 - 200</td>
                    <td style="padding: 12px 20px; color: #ef4444; font-weight: 700;">Unhealthy</td>
                    <td style="padding: 12px 20px; text-align: center;"><div style="width: 14px; height: 14px; background-color: #ef4444; border-radius: 50%; display: inline-block;"></div></td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px 20px; color: #f8fafc; font-weight: 500;">201 - 300</td>
                    <td style="padding: 12px 20px; color: #a855f7; font-weight: 700;">Very Unhealthy</td>
                    <td style="padding: 12px 20px; text-align: center;"><div style="width: 14px; height: 14px; background-color: #a855f7; border-radius: 50%; display: inline-block;"></div></td>
                </tr>
                <tr>
                    <td style="padding: 12px 20px; color: #f8fafc; font-weight: 500;">301 - 500</td>
                    <td style="padding: 12px 20px; color: #9f1239; font-weight: 700;">Hazardous</td>
                    <td style="padding: 12px 20px; text-align: center;"><div style="width: 14px; height: 14px; background-color: #9f1239; border-radius: 50%; display: inline-block;"></div></td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Failed to execute ML models or retrieve API Data. Ensure models are trained inside `models/`.")
