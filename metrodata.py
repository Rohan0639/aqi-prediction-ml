import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fallback to the provided key if not in .env
API_KEY = os.getenv("WEATHER_API_KEY", "c02ebe7947b6de1013509a77af765329")

# Station coordinates matching the ML project layout
STATIONS = {
    "Balanagar SPCB": (17.4550, 78.4483),
    "HITEC City": (17.4435, 78.3772),
    "IDA Pashamylaram SPCB": (17.5286, 78.2213),
    "Sanathnagar SPCB": (17.4569, 78.4433),
    "US Consulate": (17.4213, 78.3392),
    "Uppal SPCB": (17.4058, 78.5591),
    "Zoo Park SPCB": (17.3507, 78.4513)
}

def fetch_weather_data():
    """
    Fetches real-time weather data for all mapped stations from OpenWeatherMap.
    Matches data schema with the AQI prediction dataset.
    """
    weather_data = []
    now = datetime.now()
    
    print(f"--- Fetching Live Weather Data ({now.strftime('%Y-%m-%d %H:%M')}) ---")

    for station, (lat, lon) in STATIONS.items():
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

        try:
            response = requests.get(url)
            
            if response.status_code != 200:
                print(f"Error fetching data for {station}: HTTP {response.status_code}")
                continue
                
            data = response.json()

            temp = data.get("main", {}).get("temp")
            humidity = data.get("main", {}).get("humidity")
            pressure = data.get("main", {}).get("pressure")

            wind_speed = data.get("wind", {}).get("speed")
            wind_dir = data.get("wind", {}).get("deg")

            # OpenWeatherMap rainfall is usually under "rain" for the last 1h
            rainfall = data.get("rain", {}).get("1h", 0.0)

            weather_record = {
                "Date": now.strftime("%Y-%m-%d"),
                "Time": now.strftime("%H:%M:%S"),
                "Station": station,
                "Temperature": temp,
                "Humidity": humidity,
                "Pressure": pressure,
                "Wind_Speed": wind_speed,
                "Wind_Direction": wind_dir,
                "Rainfall": rainfall
            }
            
            weather_data.append(weather_record)
            print(f"✓ {station:<22}: {temp:>5}°C, {humidity:>3}% Humidity, Rain: {rainfall:>4}mm, Wind: {wind_speed:>5}m/s")

        except Exception as e:
            print(f"✗ Failed to fetch data for {station}: {e}")

    return weather_data

def save_weather_data_to_csv(weather_data, filename="data/hyderabad_live_weather.csv"):
    """
    Saves the fetched weather data to the specified CSV, creating directories if needed.
    """
    if not weather_data:
        print("No weather data to save.")
        return

    # Ensure the data directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df = pd.DataFrame(weather_data)
    
    # Save the file
    df.to_csv(filename, index=False)
    print(f"\nSuccessfully saved {len(df)} weather records to {filename}")

if __name__ == "__main__":
    data = fetch_weather_data()
    save_weather_data_to_csv(data)