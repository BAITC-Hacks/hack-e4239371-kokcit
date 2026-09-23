from datetime import date

import httpx
import pandas as pd

from app.config import settings

WEATHER_VARIABLES = (
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "temperature_2m",
)


def _requested_variables() -> str:
    return ",".join(
        f"{variable}_previous_day{day}" for variable in WEATHER_VARIABLES for day in (1, 2)
    )


def fetch_archived_weather(
    latitude: float,
    longitude: float,
    start_date: date | str,
    end_date: date | str,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "timezone": "Asia/Almaty",
        "hourly": _requested_variables(),
    }
    response = httpx.get(settings.open_meteo_base_url, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        payload = payload[0]
    if "hourly" not in payload:
        raise ValueError("Open-Meteo response does not contain hourly data")

    frame = pd.DataFrame(payload["hourly"])
    frame["timestamp"] = pd.to_datetime(frame.pop("time"), errors="raise")
    return frame
