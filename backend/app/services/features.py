import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "wind_10m_ms",
    "wind_100m_ms",
    "wind_100m_squared",
    "wind_100m_cubed",
    "wind_direction_sin",
    "wind_direction_cos",
    "temperature_c",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
)


def build_features(weather: pd.DataFrame, lead_days: int) -> pd.DataFrame:
    suffix = f"_previous_day{lead_days}"
    timestamp = pd.to_datetime(weather["timestamp"])
    direction = np.deg2rad(weather[f"wind_direction_100m{suffix}"])
    wind_100m = weather[f"wind_speed_100m{suffix}"] / 3.6
    hour_angle = 2 * np.pi * timestamp.dt.hour / 24
    year_angle = 2 * np.pi * timestamp.dt.dayofyear / 365.25

    return pd.DataFrame(
        {
            "wind_10m_ms": weather[f"wind_speed_10m{suffix}"] / 3.6,
            "wind_100m_ms": wind_100m,
            "wind_100m_squared": wind_100m**2,
            "wind_100m_cubed": wind_100m**3,
            "wind_direction_sin": np.sin(direction),
            "wind_direction_cos": np.cos(direction),
            "temperature_c": weather[f"temperature_2m{suffix}"],
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "day_of_year_sin": np.sin(year_angle),
            "day_of_year_cos": np.cos(year_angle),
        },
        index=weather.index,
    )
