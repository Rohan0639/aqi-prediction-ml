import requests
import json

token = '73bef76263763e96bb7672aa0ba6c5b6fca973e7'
query = 'hyderabad'
url = f"https://api.waqi.info/search/?keyword={query}&token={token}"

try:
    r = requests.get(url)
    data = r.json()
    if data['status'] == 'ok':
        stations = data['data']
        hyderabad_stations = []
        print(f"{'UID':<10} | {'Station Name'}")
        print("-" * 60)
        for s in stations:
            name = s['station']['name']
            uid = s['uid']
            url_str = s['station']['url']
            # Improved matching for Hyderabad
            if 'hyderabad' in name.lower() or 'hyderabad' in url_str.lower():
                 print(f"{uid:<10} | {name}")
                 hyderabad_stations.append({"uid": uid, "name": name})
        
        with open('hyderabad_stations.json', 'w') as f:
            json.dump(hyderabad_stations, f, indent=2)
        print(f"\nSaved {len(hyderabad_stations)} stations to hyderabad_stations.json")
except Exception as e:
    print(f"Failed to fetch stations: {e}")
