import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Key
API_KEY = os.getenv("WEATHER_API_KEY", "c02ebe7947b6de1013509a77af765329")

# Station coordinates
STATIONS = {
    "Balanagar SPCB": (17.4550, 78.4483),
    "HITEC City": (17.4435, 78.3772),
    "IDA Pashamylaram SPCB": (17.5286, 78.2213),
    "Sanathnagar SPCB": (17.4569, 78.4433),
    "US Consulate": (17.4213, 78.3392),
    "Uppal SPCB": (17.4058, 78.5591),
    "Zoo Park SPCB": (17.3507, 78.4513)
}

# Convert wind degree → direction label
def get_wind_direction_label(deg):
    if deg is None:
        return "NA"
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((deg + 22.5) // 45) % 8
    return directions[idx]

def fetch_weather_data():
    weather_data = []
    now = datetime.now()
    
    print(f"\n--- Fetching Live Weather Data ({now.strftime('%Y-%m-%d %H:%M')}) ---")

    for station, (lat, lon) in STATIONS.items():
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

        try:
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                print(f"❌ {station}: HTTP {response.status_code}")
                continue

            data = response.json()

            main = data.get("main", {})
            wind = data.get("wind", {})

            temp = main.get("temp")
            humidity = main.get("humidity")
            pressure = main.get("pressure")

            wind_speed = wind.get("speed")
            wind_dir = wind.get("deg")
            wind_dir_label = get_wind_direction_label(wind_dir)

            rainfall = data.get("rain", {}).get("1h", 0.0)

            record = {
                "Date": now.strftime("%Y-%m-%d"),
                "Time": now.strftime("%H:%M:%S"),
                "Station": station,
                "Temperature": temp,
                "Humidity": humidity,
                "Pressure": pressure,
                "Wind_Speed": wind_speed,
                "Wind_Direction_Deg": wind_dir,
                "Wind_Direction_Label": wind_dir_label,
                "Rainfall": rainfall
            }

            weather_data.append(record)

            # Safe print
            print(
                f"✓ {station:<22}: "
                f"{temp if temp is not None else 'NA'}°C | "
                f"{humidity if humidity is not None else 'NA'}% | "
                f"Wind: {wind_speed if wind_speed is not None else 'NA'} m/s "
                f"({wind_dir if wind_dir is not None else 'NA'}° {wind_dir_label}) | "
                f"Rain: {rainfall} mm"
            )

        except Exception as e:
            print(f"❌ Failed {station}: {e}")

    return weather_data


def save_weather_data_to_csv(weather_data, filename="data/hyderabad_live_weather.csv"):
    if not weather_data:
        print("⚠ No weather data to save.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df_new = pd.DataFrame(weather_data)

    # If file exists → merge + update
    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)

        # Combine
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        # Remove duplicates → keep latest (new replaces old)
        df_combined = df_combined.drop_duplicates(
            subset=["Date", "Station"], keep="last"
        )

        print("🔄 Updated dataset (no duplicates, latest values kept)")

    else:
        df_combined = df_new
        print("📁 Created new dataset")

    # Save
    df_combined.to_csv(filename, index=False)

    print(f"\n✅ Dataset saved: {filename}")
    print(f"📊 Total rows: {len(df_combined)}")


if __name__ == "__main__":
    data = fetch_weather_data()
    save_weather_data_to_csv(data)