"""External API clients: OpenWeather + NHTSA vPIC."""

import os
import requests
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5"
NHTSA_VIN_BASE = "https://vpic.nhtsa.dot.gov/api/vehicles"
NHTSA_RECALLS_BASE = "https://api.nhtsa.gov/recalls/recallsByVehicle"


# ---------------------------------------------------------------------------
# OpenWeather client
# ---------------------------------------------------------------------------

def fetch_weather(city: str) -> Optional[Dict[str, Any]]:
    """
    Fetch current weather for a city.
    Returns a dict with temperature (°F), precipitation (in/hr), visibility (miles),
    description, humidity, wind_speed (mph) — or None on failure.

    Notes on OpenWeather imperial mode:
      - temperature  → °F  (via units=imperial)
      - wind_speed   → mph (via units=imperial)
      - visibility   → always metres regardless of units; converted here to miles
      - precipitation→ always mm regardless of units; converted here to inches
    """
    try:
        url = f"{OPENWEATHER_BASE}/weather"
        params = {
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "imperial",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Precipitation is always mm from OW — convert to inches
        rain_mm = 0.0
        if "rain" in data:
            rain_mm = data["rain"].get("1h", data["rain"].get("3h", 0.0))
        elif "snow" in data:
            rain_mm = data["snow"].get("1h", data["snow"].get("3h", 0.0))
        rain_in = rain_mm * 0.0393701

        # Visibility is always metres from OW — convert to miles
        visibility_miles = data.get("visibility", 16093) / 1609.34

        return {
            "city": data.get("name", city),
            "temperature": round(data["main"]["temp"], 1),       # °F
            "precipitation": round(rain_in, 3),                  # in/hr
            "visibility": round(visibility_miles, 2),            # miles
            "description": data["weather"][0]["description"].title(),
            "humidity": data["main"]["humidity"],
            "wind_speed": round(data["wind"]["speed"], 1),       # mph
        }
    except requests.exceptions.HTTPError as exc:
        logger.warning("OpenWeather HTTP error for '%s': %s", city, exc)
        return None
    except Exception as exc:
        logger.error("OpenWeather fetch failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# NHTSA vPIC + Recalls client
# ---------------------------------------------------------------------------

def decode_vin(vin: str) -> Optional[Dict[str, Any]]:
    """
    Decode a VIN using NHTSA vPIC API.
    Returns make, model, year, vehicle_type — or None on failure.
    """
    vin = vin.strip().upper()
    try:
        url = f"{NHTSA_VIN_BASE}/DecodeVinValues/{vin}"
        resp = requests.get(url, params={"format": "json"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("Results", [{}])[0]

        make = results.get("Make", "Unknown")
        model = results.get("Model", "Unknown")
        year_str = results.get("ModelYear", "0")
        vehicle_type = results.get("VehicleType", "Unknown")
        error_code = results.get("ErrorCode", "0")

        # ErrorCode 0 = no error; 1 = check digit mismatch but still parsed
        if error_code not in ("0", "1", "6"):
            logger.warning("VIN %s decode returned error code %s", vin, error_code)

        try:
            year = int(year_str)
        except (ValueError, TypeError):
            year = 0

        return {
            "vin": vin,
            "make": make or "Unknown",
            "model": model or "Unknown",
            "year": year,
            "vehicle_type": vehicle_type,
        }
    except Exception as exc:
        logger.error("VIN decode failed for %s: %s", vin, exc)
        return None


def fetch_recalls(make: str, model: str, year: int) -> list:
    """
    Fetch active NHTSA safety recalls for a given make/model/year.
    Returns a list of recall dicts.
    """
    try:
        params = {"make": make, "model": model, "modelYear": year}
        resp = requests.get(NHTSA_RECALLS_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        return [
            {
                "recall_id": r.get("NHTSACampaignNumber", ""),
                "component": r.get("Component", ""),
                "summary": r.get("Summary", ""),
                "consequence": r.get("Consequence", ""),
                "remedy": r.get("Remedy", ""),
                "report_date": r.get("ReportReceivedDate", ""),
            }
            for r in results
        ]
    except Exception as exc:
        logger.error("Recalls fetch failed for %s %s %s: %s", make, model, year, exc)
        return []


def full_vin_lookup(vin: str) -> Optional[Dict[str, Any]]:
    """Decode VIN then fetch recalls. Returns a combined dict."""
    vehicle = decode_vin(vin)
    if not vehicle:
        return None

    recalls = []
    if vehicle["year"] > 0 and vehicle["make"] != "Unknown":
        recalls = fetch_recalls(vehicle["make"], vehicle["model"], vehicle["year"])

    vehicle["has_recall"] = len(recalls) > 0
    vehicle["recall_count"] = len(recalls)
    vehicle["recalls"] = recalls

    if recalls:
        vehicle["recall_description"] = "; ".join(
            r["component"] for r in recalls[:3]
        )
    else:
        vehicle["recall_description"] = None

    return vehicle
