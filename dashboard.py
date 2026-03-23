import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
    initial_sidebar_state="expanded"
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
    /* Global Fonts & Background Injection */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #082f49 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* Hide Streamlit default UI elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    header[data-testid="stHeader"] {background: transparent;}
    footer {visibility: hidden;}

    /* Glassmorphism Classes */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 1.5rem;
        transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1), background 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        background: rgba(30, 41, 59, 0.6);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Premium Title */
    .premium-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(to right, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    
    .premium-subtitle {
        font-size: 1.2rem;
        color: #3b82f6;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* Metric Layouts */
    .metric-flex {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        color: #f8fafc;
    }
    .metric-value.large {
        font-size: 4.5rem;
    }
    .metric-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* Live Badge */
    .live-badge-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        margin-bottom: 2rem;
    }
    .live-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 6px 16px;
        border-radius: 9999px;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
    }
    .pulse-dot-premium {
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 12px;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.8);
        animation: pulsePremium 2s infinite cubic-bezier(0.4, 0, 0.6, 1);
    }
    @keyframes pulsePremium {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 2rem 0 1.5rem 0;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        white-space: nowrap;
    }
    .section-line {
        height: 1px;
        flex-grow: 1;
        background: linear-gradient(to right, rgba(59, 130, 246, 0.5), transparent);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
    }
    
    /* Custom Scrollbar for a sleek look */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
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
# TREND DATA & CHART HELPERS
# ---------------------------------------------------------
LIVE_AQI_CSV = 'data/live_aqi_dataset.csv'
HISTORICAL_CSV = 'data/hyderabad_air_quality_10y_combined_fixed.csv'

@st.cache_data(ttl=120)
def load_trend_data(station_name, days=7):
    """Load the last N days of AQI + Temperature data for a station from combined sources."""
    dfs = []
    
    if os.path.exists(HISTORICAL_CSV):
        try:
            dfs.append(pd.read_csv(HISTORICAL_CSV, parse_dates=['Date']))
        except Exception:
            pass
            
    if os.path.exists(LIVE_AQI_CSV):
        try:
            dfs.append(pd.read_csv(LIVE_AQI_CSV, parse_dates=['Date']))
        except Exception:
            pass
            
    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df[df['Station'] == station_name].copy()
    if df.empty:
        return df

    # Drop duplicates keeping the latest ones
    df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
    
    df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
    df = df.dropna(subset=['Date', 'AQI'])
    
    # Take the last N available days (avoids empty charts if live data falls behind)
    df = df.tail(days).reset_index(drop=True)
    return df[['Date', 'AQI', 'Temperature']]


def render_aqi_trend_chart(df):
    """Plotly line chart for AQI Trend with AQI-level colored markers."""
    if df.empty or df['AQI'].dropna().empty:
        st.info("Not enough AQI data available for this station to render the trend.")
        return

    df = df.dropna(subset=['AQI']).copy()
    df['Color'] = df['AQI'].apply(get_aqi_category_color)
    df['Category'] = df['AQI'].apply(get_aqi_category)
    df['DateStr'] = df['Date'].dt.strftime('%b %d, %Y')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['AQI'], mode='lines',
        line=dict(color='rgba(59,130,246,0.15)', width=0),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.06)',
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['AQI'], mode='lines+markers',
        line=dict(color='#3b82f6', width=3, shape='spline'),
        marker=dict(size=12, color=df['Color'],
                    line=dict(width=2, color='#0f172a'), symbol='circle'),
        customdata=list(zip(df['DateStr'], df['AQI'].astype(int), df['Category'])),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'AQI: <b>%{customdata[1]}</b><br>'
            'Category: %{customdata[2]}<extra></extra>'
        ),
        name='AQI',
    ))

    bands = [
        (0, 50, '#22c55e'), (50, 100, '#eab308'), (100, 150, '#f97316'),
        (150, 200, '#ef4444'), (200, 300, '#a855f7'),
    ]
    max_aqi = max(df['AQI'].max() * 1.15, 60)
    for lo, hi, color in bands:
        if lo < max_aqi:
            fig.add_hrect(y0=lo, y1=min(hi, max_aqi),
                          fillcolor=color, opacity=0.06,
                          line_width=0, layer='below')

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.5)',
        margin=dict(l=20, r=20, t=10, b=20), height=370,
        xaxis=dict(showgrid=False, tickformat='%b %d',
                   tickfont=dict(size=12, color='#94a3b8')),
        yaxis=dict(title='AQI', title_font=dict(size=13, color='#94a3b8'),
                   gridcolor='rgba(148,163,184,0.1)',
                   tickfont=dict(size=12, color='#94a3b8'), rangemode='tozero'),
        hoverlabel=dict(bgcolor='#1e293b', font_size=13,
                        font_color='#f1f5f9', bordercolor='#334155'),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_temperature_trend_chart(df):
    """Plotly line chart for Temperature Trend with warm orange styling."""
    if df.empty or df['Temperature'].dropna().empty:
        st.info("Not enough Temperature data available for this station to render the trend.")
        return

    df = df.dropna(subset=['Temperature']).copy()
    df['DateStr'] = df['Date'].dt.strftime('%b %d, %Y')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Temperature'], mode='lines',
        line=dict(color='rgba(249,115,22,0.15)', width=0),
        fill='tozeroy', fillcolor='rgba(249,115,22,0.08)',
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Temperature'], mode='lines+markers',
        line=dict(color='#f97316', width=3, shape='spline'),
        marker=dict(size=12, color='#f97316',
                    line=dict(width=2, color='#0f172a'), symbol='circle'),
        customdata=list(zip(df['DateStr'], df['Temperature'].round(1))),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'Temperature: <b>%{customdata[1]} °C</b><extra></extra>'
        ),
        name='Temperature',
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.5)',
        margin=dict(l=20, r=20, t=10, b=20), height=370,
        xaxis=dict(showgrid=False, tickformat='%b %d',
                   tickfont=dict(size=12, color='#94a3b8')),
        yaxis=dict(title='Temperature (°C)', title_font=dict(size=13, color='#94a3b8'),
                   gridcolor='rgba(148,163,184,0.1)',
                   tickfont=dict(size=12, color='#94a3b8')),
        hoverlabel=dict(bgcolor='#1e293b', font_size=13,
                        font_color='#f1f5f9', bordercolor='#334155'),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# UI COMPONENTS HELPERS
# ---------------------------------------------------------

def render_glass_metric(label, value, category, color, icon="", confidence=None, model=None, sub_text=None, border_color=None):
    if not border_color: border_color = color
    
    conf_html = ""
    if confidence is not None and confidence != '--':
        try:
            c_val = float(confidence)
            conf_color = "#22c55e" if c_val >= 90 else "#eab308" if c_val >= 80 else "#ef4444"
            conf_html = f'<div style="color: {conf_color}; font-size: 0.85rem; background: {conf_color}15; padding: 2px 8px; border-radius: 6px; font-weight: 600;">🎯 {c_val}%</div>'
        except:
            pass
        
    model_html = f'<div style="color: #60a5fa; font-size: 0.85rem;">{model}</div>' if model and model != '--' and model != 'Model: --' else ""
    sub_html = f'<div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;">{sub_text}</div>' if sub_text else ""
    
    flex_div = ""
    if model_html or conf_html:
        flex_div = f'<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">{model_html}{conf_html}</div>'

    html_content = f'<div class="glass-card" style="border-left: 4px solid {border_color};">'
    html_content += f'<div class="metric-label">{icon} {label}</div>'
    html_content += f'<div class="metric-value" style="color: {color};">{value}</div>'
    html_content += f'<div class="metric-tag" style="background-color: {color}15; color: {color}; border: 1px solid {color}30;">{category}</div>'
    if flex_div:
        html_content += flex_div
    if sub_html:
        html_content += sub_html
    html_content += '</div>'
    
    st.markdown(html_content, unsafe_allow_html=True)

def render_section_header(title, icon=""):
    st.markdown(f"""
    <div class="section-header">
        <h2 class="section-title">{icon} {title}</h2>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# RENDER UI
# ---------------------------------------------------------

# Header
st.markdown(f"""
<div style="text-align: center; padding: 2rem 0;">
    <h1 class="premium-title">Hyderabad AQI Dashboard</h1>
    <div class="premium-subtitle">Real-Time Forecasts & Insights</div>
</div>
<div class="live-badge-wrapper">
    <div class="live-badge">
        <div class="pulse-dot-premium"></div>
        <span style="color: #10b981; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase;">
            LIVE SYSTEM STATUS <span style="color: #6ee7b7; font-weight: 400; margin-left: 8px;">Last Updated: {datetime.now().strftime('%H:%M:%S')}</span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
<div style="padding: 1rem 0; text-align: center;">
    <h2 style="font-weight: 800; font-size: 1.5rem; margin-bottom: 0;">AQI CONTROL PANEL</h2>
    <div style="color: #3b82f6; font-size: 0.8rem; text-transform: uppercase;"></div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["Dashboard", "Hyderabad Station Map", "AQI Categories Info"], label_visibility="collapsed")

st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
st.sidebar.caption("Powered by Wait-for-it Model • v2.0 Premium")

with st.spinner("Executing Models and Fetching WAQI Live Environmental Data..."):
    all_station_details, stations_current, stations_prediction = get_dashboard_data()

if page == "Hyderabad Station Map":
    render_map_view(all_station_details)
    st.stop()
    
elif page == "AQI Categories Info":
    render_section_header("AQI Health & Categories Guide", "📖")
    st.markdown("""
    <div class="glass-card" style="padding: 2.5rem; text-align: center;">
        <h3 style="font-size: 2rem; font-weight: 800; margin-bottom: 2rem;">Air Quality Index Reference</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; text-align: left;">
            <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); padding: 1.5rem; border-radius: 12px; transition: transform 0.2s; cursor: default;" onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
                <h4 style="color: #22c55e; margin: 0 0 0.5rem 0; font-size: 1.25rem;">● Good (0 - 50)</h4>
                <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; margin:0;">Air quality is considered satisfactory, and air pollution poses little or no risk.</p>
            </div>
            <div style="background: rgba(234, 179, 8, 0.1); border: 1px solid rgba(234, 179, 8, 0.3); padding: 1.5rem; border-radius: 12px; transition: transform 0.2s; cursor: default;" onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
                <h4 style="color: #eab308; margin: 0 0 0.5rem 0; font-size: 1.25rem;">● Moderate (51 - 100)</h4>
                <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; margin:0;">Air quality is acceptable; however, there may be a moderate health concern for a very small number of people.</p>
            </div>
            <div style="background: rgba(249, 115, 22, 0.1); border: 1px solid rgba(249, 115, 22, 0.3); padding: 1.5rem; border-radius: 12px; transition: transform 0.2s; cursor: default;" onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
                <h4 style="color: #f97316; margin: 0 0 0.5rem 0; font-size: 1.25rem;">● Unhealthy for Sensitive Groups (101 - 150)</h4>
                <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; margin:0;">Members of sensitive groups may experience health effects. The general public is not likely to be affected.</p>
            </div>
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); padding: 1.5rem; border-radius: 12px; transition: transform 0.2s; cursor: default;" onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
                <h4 style="color: #ef4444; margin: 0 0 0.5rem 0; font-size: 1.25rem;">● Unhealthy (151 - 200)</h4>
                <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; margin:0;">Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.</p>
            </div>
            <div style="background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.3); padding: 1.5rem; border-radius: 12px; transition: transform 0.2s; cursor: default;" onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
                <h4 style="color: #a855f7; margin: 0 0 0.5rem 0; font-size: 1.25rem;">● Very Unhealthy (201 - 300)</h4>
                <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; margin:0;">Health alert: everyone may experience more serious health effects.</p>
            </div>
            <div style="background: rgba(159, 18, 57, 0.1); border: 1px solid rgba(159, 18, 57, 0.3); padding: 1.5rem; border-radius: 12px; transition: transform 0.2s; cursor: default;" onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
                <h4 style="color: #9f1239; margin: 0 0 0.5rem 0; font-size: 1.25rem;">● Hazardous (301 - 500)</h4>
                <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; margin:0;">Health warnings of emergency conditions. The entire population is more likely to be affected.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if all_station_details:
    # --- CALCULATE CITY AVERAGE AQI ---
    valid_current_aqi = [data['current_aqi'] for data in all_station_details.values() if isinstance(data['current_aqi'], (int, float))]
    valid_pred_aqi = [data['predicted_aqi'] for data in all_station_details.values() if isinstance(data['predicted_aqi'], (int, float))]
    
    avg_current_aqi = int(sum(valid_current_aqi) / len(valid_current_aqi)) if valid_current_aqi else None
    avg_pred_aqi = int(sum(valid_pred_aqi) / len(valid_pred_aqi)) if valid_pred_aqi else None

    # City Averages UI
    render_section_header("Hyderabad City Summary", "🏙️")
    avg_col1, avg_col2 = st.columns(2, gap="large")
    
    with avg_col1:
        cat_curr = get_aqi_category(avg_current_aqi) if avg_current_aqi else '--'
        color_curr = get_aqi_category_color(avg_current_aqi) if avg_current_aqi else 'gray'
        render_glass_metric(
            label="AVERAGE LIVE AQI",
            value=avg_current_aqi if avg_current_aqi else '--',
            category=cat_curr,
            color=color_curr,
            icon=""
        )

    with avg_col2:
        cat_pred = get_aqi_category(avg_pred_aqi) if avg_pred_aqi else '--'
        color_pred = get_aqi_category_color(avg_pred_aqi) if avg_pred_aqi else 'gray'
        
        trend_arrow = ""
        trend_text = ""
        if avg_current_aqi and avg_pred_aqi:
            if avg_pred_aqi > avg_current_aqi:
                trend_arrow = "↑"
                trend_text = "Expected to worsen"
                t_color = "#ef4444"
            elif avg_pred_aqi < avg_current_aqi:
                trend_arrow = "↓"
                trend_text = "Improvement expected"
                t_color = "#22c55e"
            else:
                trend_arrow = "→"
                trend_text = "Staying stable"
                t_color = "#9ca3af"

        sub_html = f"<span style='color:{t_color};'><b style='font-size:1.1rem;'>{trend_arrow}</b> {trend_text}</span>" if trend_text else ""
        
        render_glass_metric(
            label="FORECASTED AVERAGE (24H)",
            value=f"{avg_pred_aqi} {trend_arrow}" if avg_pred_aqi else '--',
            category=cat_pred,
            color=color_pred,
            icon="",
            sub_text=sub_html
        )

    # --- STATION SELECTION ---
    render_section_header("SELECT MONITORING SYSTEM 🔗", "")
    
    station_names = list(all_station_details.keys())
    clean_to_orig = {clean_station_name(name): name for name in station_names}
    
    selected_clean_name = st.selectbox(
        "Select Monitoring Node", 
        options=list(clean_to_orig.keys()), 
        label_visibility="collapsed"
    )
    
    selected_station = clean_to_orig[selected_clean_name]
    station_data = all_station_details[selected_station]
    
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(f"### Current AQI ({selected_clean_name})")
        aqi_val = station_data.get('current_aqi', '--')
        category = f"Category: {get_aqi_category(aqi_val)}" if isinstance(aqi_val, (int, float)) else '--'
        pollutant = station_data.get('dominant_pollutant', '--')
        color = get_aqi_category_color(aqi_val if isinstance(aqi_val, (int, float)) else None)
        
        render_glass_metric(
            label=f"LIVE API READING",
            value=aqi_val,
            category=category,
            color=color,
            icon="",
            sub_text=f"Dominant Pollutant: {pollutant}",
            border_color="#f97316"
        )

    with col2:
        st.markdown(f"### Tomorrow AQI Prediction")
        pred_aqi = station_data.get('predicted_aqi', '--')
        pred_cat = f"Category: {get_aqi_category(pred_aqi)}" if isinstance(pred_aqi, (int, float)) else '--'
        model_name = "Model: " + station_data.get('model_used', '--')
        confidence = station_data.get('confidence_score', '--')
        color = get_aqi_category_color(pred_aqi if isinstance(pred_aqi, (int, float)) else None)
        
        render_glass_metric(
            label="AI FORECAST",
            value=pred_aqi,
            category=pred_cat,
            color=color,
            icon="",
            model=model_name,
            confidence=confidence,
            border_color="#f97316"
        )


    # --- WEATHER SECTION (CSV) ---
    render_section_header("Live Weather Conditions", "🌬️")
    st.markdown(f"<p style='color: #94a3b8; font-size: 0.9rem; margin-top: -1.5rem; margin-bottom: 1.5rem;'>📍 Showing weather for: {selected_station}</p>", unsafe_allow_html=True)
    
    _wx_map = load_weather_from_csv()
    _wx_data = get_csv_weather_for_station(selected_station, _wx_map)

    _temperature = _wx_data.get('temperature', '--')
    _humidity = _wx_data.get('humidity', '--')
    _wind_speed = _wx_data.get('wind_speed', '--')
    _wind_label = _wx_data.get('wind_dir_label', '--')
    _wind_deg = _wx_data.get('wind_dir_deg', '--')
    _wind_icon = WIND_ICON_MAP.get(_wind_label, '🧭')

    wx_c1, wx_c2 = st.columns(2, gap="medium")
    wx_c3, wx_c4 = st.columns(2, gap="medium")
    
    weather_card_html = """
    <div class="glass-card" style="text-align: center; padding: 1.5rem 1rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.25rem; font-weight: 600;">{label}</div>
        <div style="font-size: 1.75rem; font-weight: 800; color: #f8fafc;">{value}</div>
    </div>
    """
    
    with wx_c1:
        st.markdown(weather_card_html.format(border="#f97316", icon="🌡️", label="Temperature", value=f"{_temperature} °C" if _temperature != '--' else '--'), unsafe_allow_html=True)
    with wx_c2:
        st.markdown(weather_card_html.format(border="#38bdf8", icon="💧", label="Humidity", value=f"{_humidity} %" if _humidity != '--' else '--'), unsafe_allow_html=True)
    with wx_c3:
        st.markdown(weather_card_html.format(border="#34d399", icon="💨", label="Wind Speed", value=f"{_wind_speed} m/s" if _wind_speed != '--' else '--'), unsafe_allow_html=True)
    with wx_c4:
        st.markdown(weather_card_html.format(border="#9ca3af", icon="🧭", label="Wind Direction", value=f"<span style='font-size: 1.75rem; vertical-align: middle;'>{_wind_icon}</span> <span style='font-size: 1.25rem; vertical-align: middle; color: #34d399;'>{_wind_label} ({_wind_deg}°)</span>" if _wind_label != '--' else '--'), unsafe_allow_html=True)

    # --- 7-DAY TREND CHARTS ---
    render_section_header(f"Historical Trends of {selected_clean_name}", "📈")
    
    trend_df = load_trend_data(selected_station, days=7)
    
    st.markdown('<div class="glass-card" style="padding: 1rem 1.5rem 0.5rem 1.5rem; border-top: 2px solid #a855f7;">', unsafe_allow_html=True)
    render_aqi_trend_chart(trend_df)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("Failed to execute ML models or retrieve API Data. Ensure models are trained inside `models/`.")
