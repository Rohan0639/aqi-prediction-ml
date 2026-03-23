import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import pandas as pd

def get_aqi_category_color(aqi):
    if aqi is None: return "gray"
    if aqi <= 50: return "#22c55e" # Green
    if aqi <= 100: return "#eab308" # Yellow
    if aqi <= 150: return "#f97316" # Orange
    if aqi <= 200: return "#ef4444" # Red
    if aqi <= 300: return "#a855f7" # Purple
    return "#9f1239" # Maroon

def get_aqi_category(aqi):
    if aqi is None: return "Unknown"
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

def inject_styles():
    st.markdown("""
        <style>
        div[data-testid="stMetricValue"] {
            font-size: 2.5rem !important;
            font-weight: 800 !important;
        }
        .main-metric-container {
            padding: 1.25rem;
            border-radius: 12px;
            background-color: #111827; 
            border: 1px solid #1f2937;
            margin-bottom: 1rem;
        }
        .metric-title {
            color: #9ca3af;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .metric-value {
            font-size: 3rem;
            font-weight: 900;
            line-height: 1;
            margin-bottom: 0.5rem;
        }
        .metric-category {
            font-size: 1.15rem;
            font-weight: 700;
        }
        </style>
    """, unsafe_allow_html=True)

def render_map_view(all_station_details):
    inject_styles()
    st.title("Hyderabad AQI Monitoring Stations")
    st.markdown("### Interactive Geographic AQI Forecast")
    
    # Define station coordinates
    stations_coords = {
        "Balanagar SPCB": {"lat": 17.4589, "lon": 78.4412},
        "HITEC City": {"lat": 17.4419, "lon": 78.3801}, 
        "IDA Pashamylaram SPCB": {"lat": 17.5303, "lon": 78.1820},
        "Sanathnagar SPCB": {"lat": 17.4561, "lon": 78.4437},
        "US Consulate": {"lat": 17.4170, "lon": 78.3470}, 
        "Uppal SPCB": {"lat": 17.4018, "lon": 78.5602},
        "Zoo Park SPCB": {"lat": 17.3507, "lon": 78.4432}
    }

    # Helper function for Folium standard colors
    def get_folium_color(aqi):
        if aqi is None: return "gray"
        if aqi <= 50: return "green"
        if aqi <= 100: return "yellow"
        if aqi <= 150: return "orange"
        if aqi <= 200: return "red"
        if aqi <= 300: return "purple"
        return "darkred"

    # Create map centered on Hyderabad
    m = folium.Map(location=[17.4000, 78.4000], zoom_start=11, tiles="CartoDB dark_matter")
    
    # Add Current Location Button
    plugins.LocateControl(position="topleft", strings={"title": "Show my current location", "popup": "You are here"}).add_to(m)

    # Add markers with styled tooltips
    for station_name, details in all_station_details.items():
        if station_name in stations_coords:
            coord = stations_coords[station_name]
            aqi = details.get('current_aqi', 0)
            color = get_folium_color(aqi)
            category = get_aqi_category(aqi) if isinstance(aqi, (int, float)) else '--'
            hex_color = get_aqi_category_color(aqi) if isinstance(aqi, (int, float)) else 'gray'

            # Clean station name for display
            display_name = station_name.replace(' SPCB', '').replace(',', '').strip()

            # Rich HTML tooltip shown on hover
            tooltip_html = f"""
            <div style="font-family: 'Segoe UI', Arial, sans-serif; padding: 4px 6px; min-width: 140px;">
                <b style="font-size: 13px;">{display_name}</b><br>
                <span style="color: {hex_color}; font-weight: 700; font-size: 12px;">AQI: {aqi} — {category}</span>
            </div>
            """

            folium.CircleMarker(
                location=[coord['lat'], coord['lon']],
                radius=10,
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=station_name,  # Used for click detection by st_folium
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
            ).add_to(m)

    # Display map and capture output
    map_data = st_folium(m, width=700, height=380, key="hyderabad_aqi_map")

    # Handle Click Interaction
    selected_from_map = None
    if map_data and map_data.get("last_object_clicked_popup"):
        selected_from_map = map_data["last_object_clicked_popup"]

    if selected_from_map and selected_from_map in all_station_details:
        st.markdown(f"## Station Details: {selected_from_map}")
        station_data = all_station_details[selected_from_map]
        
        # Display simplified metrics (similar to dashboard)
        col1, col2 = st.columns(2)
        
        with col1:
            curr_aqi = station_data.get('current_aqi', '--')
            curr_cat = get_aqi_category(curr_aqi) if isinstance(curr_aqi, (int, float)) else '--'
            curr_color = get_aqi_category_color(curr_aqi if isinstance(curr_aqi, (int, float)) else None)
            
            st.markdown(f"""
            <div class="main-metric-container" style="border-left: 6px solid {curr_color};">
                <div class="metric-title">Current Live AQI</div>
                <div class="metric-value" style="color: {curr_color};">{curr_aqi}</div>
                <div class="metric-category" style="color: {curr_color};">{curr_cat}</div>
                <div style="color: #d1d5db; margin-top: 0.5rem;">Pollutant: {station_data.get('dominant_pollutant', '--')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            pred_aqi = station_data.get('predicted_aqi', '--')
            pred_cat = get_aqi_category(pred_aqi) if isinstance(pred_aqi, (int, float)) else '--'
            pred_color = get_aqi_category_color(pred_aqi if isinstance(pred_aqi, (int, float)) else None)
            confidence = station_data.get('confidence_score', '--')
            
            st.markdown(f"""
            <div class="main-metric-container" style="border-left: 6px solid {pred_color};">
                <div class="metric-title">Tomorrow AI Forecast</div>
                <div class="metric-value" style="color: {pred_color};">{pred_aqi}</div>
                <div class="metric-category" style="color: {pred_color};">{pred_cat}</div>
                <div style="color: #60a5fa; margin-top: 0.5rem; font-weight: 500;">🎯 Confidence: {confidence}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Weather info
        if 'weather' in station_data:
            w = station_data['weather']
            w_col1, w_col2, w_col3 = st.columns(3)
            w_col1.metric("Temperature", f"{w.get('temperature', '--')} °C")
            w_col2.metric("Humidity", f"{w.get('humidity', '--')} %")
            w_col3.metric("Rainfall", f"{w.get('rain', '--')} mm")
    else:
        st.info("💡 Click on a marker to view detailed station metrics and environmental forecasts.")

    # Legend
    st.markdown("---")
    st.markdown("### AQI Color Legend")
    l_cols = st.columns(6)
    legend_items = [
        ("Good (0-50)", "#22c55e"),
        ("Satisfactory (51-100)", "#eab308"),
        ("Moderate (101-200)", "#f97316"),
        ("Poor (201-300)", "#ef4444"),
        ("Very Poor (301-400)", "#a855f7"),
        ("Severe (401+)", "#9f1239")
    ]
    for i, (text, color) in enumerate(legend_items):
        with l_cols[i % 6]:
            html = f'''
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem; justify-content: center;">
                <div style="min-width: 14px; min-height: 14px; max-width: 14px; max-height: 14px; background-color: {color}; border-radius: 50%; margin-right: 8px;"></div>
                <span style="font-size: 0.85rem; font-weight: 600; color: #f8fafc; line-height: 1.2;">{text.replace(" (", "<br>(")}</span>
            </div>
            '''
            st.markdown(html, unsafe_allow_html=True)
