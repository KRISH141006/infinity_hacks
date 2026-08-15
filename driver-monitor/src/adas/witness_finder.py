# witness_finder.py - Geofence Connected Vehicle Witness Discovery
import urllib.request
import json
import random
from datetime import datetime

class GoogleMapsWitnessFinder:
    """Simulates a geofence-based witness discovery platform.
    
    In connected vehicle ADAS infrastructure, this queries telematic databases
    for nearby connected vehicles around the accident GPS coordinates & timestamp.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.simulated_traffic_pool = [
            {"model": "Tesla Model Y", "color": "Pearl White", "heading": "Northbound", "range_m": 42},
            {"model": "Toyota RAV4", "color": "Magnetic Gray", "heading": "Northbound", "range_m": 85},
            {"model": "Ford F-150", "color": "Shadow Black", "heading": "Southbound", "range_m": 12},
            {"model": "Honda Civic", "color": "Rallye Red", "heading": "Northbound", "range_m": 60},
            {"model": "Chevrolet Tahoe", "color": "Summit White", "heading": "Southbound", "range_m": 95},
            {"model": "BMW 3-Series", "color": "Alpine White", "heading": "Northbound", "range_m": 110},
            {"model": "Freightliner Semi-Truck", "color": "Dark Blue", "heading": "Southbound", "range_m": 35}
        ]

    def reverse_geocode(self, lat, lon):
        """Simulates calling Google Maps Geocoding API to get exact street address."""
        if self.api_key:
            try:
                url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={self.api_key}"
                with urllib.request.urlopen(url, timeout=3) as response:
                    data = json.loads(response.read().decode())
                    if data.get("status") == "OK" and len(data.get("results")) > 0:
                        return data["results"][0]["formatted_address"]
            except Exception as e:
                print(f"[Witness Finder] API error: {e}, falling back to default geocoding.")
                
        return "Highway 101 Northbound, Mile Marker 18.2, Silicon Valley, CA"

    def discover_nearby_witness_vehicles(self, lat, lon, timestamp=None):
        """Queries the telematic database to locate nearby vehicles that witnessed an event."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        address = self.reverse_geocode(lat, lon)
        num_witnesses = random.randint(2, 4)
        discovered_witnesses = random.sample(self.simulated_traffic_pool, min(num_witnesses, len(self.simulated_traffic_pool)))
        discovered_witnesses = sorted(discovered_witnesses, key=lambda x: x["range_m"])
        
        witness_report = {
            "accident_address": address,
            "gps_coordinates": {"latitude": lat, "longitude": lon},
            "timestamp": timestamp,
            "witness_discovery_status": "SUCCESS",
            "witness_count": len(discovered_witnesses),
            "witness_list": [
                {
                    "anonymous_device_id": f"DEV-{random.randint(100000, 999999)}-VEH",
                    "vehicle_description": f"{w['color']} {w['model']}",
                    "direction_of_travel": w["heading"],
                    "proximity_distance_meters": w["range_m"],
                    "camera_coverage_probability": "High" if w["range_m"] < 50 else "Medium",
                    "request_status": "Pending Verification"
                }
                for w in discovered_witnesses
            ],
            "legal_subpoena_procedures": [
                "1. Log Geofence coordinate timestamp into EDR blackbox registry.",
                "2. Dispatch automated ping to OEM connected cloud for identified witness device hashes.",
                "3. Request encrypted 30-second rolling dashcam cache from nearby witness nodes."
            ]
        }
        return witness_report
