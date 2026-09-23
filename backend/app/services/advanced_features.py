"""Candidate forecast-only features; never read SCADA observations here."""

import numpy as np
import pandas as pd

from app.services.features import build_context_features


def build_advanced_features(weather: pd.DataFrame, lead_days: int, trajectory: bool = False):
    result = build_context_features(weather, lead_days)
    timestamp = pd.to_datetime(weather.timestamp)
    groups = timestamp.dt.date
    suffix = f"_previous_day{lead_days}"
    direction = np.deg2rad(weather[f"wind_direction_100m{suffix}"])
    additions = {
        "shear_ratio": result.wind_100m_ms / (result.wind_10m_ms + 0.5),
        "density_adjusted_wind": result.wind_100m_ms
        * np.cbrt(288.15 / (result.temperature_c + 273.15)),
        "direction_sin_2": np.sin(2 * direction),
        "direction_cos_2": np.cos(2 * direction),
        "direction_sin_3": np.sin(3 * direction),
        "direction_cos_3": np.cos(3 * direction),
    }
    for column in ("wind_u", "wind_v", "wind_shear"):
        grouped = result[column].groupby(groups)
        for operation in ("mean", "min", "max", "std"):
            additions[f"{column}_daily_{operation}"] = grouped.transform(operation)
        for offset in (-9, -3, 3, 9):
            additions[f"{column}_offset_{offset}"] = grouped.shift(offset).fillna(result[column])
    for column in ("wind_100m_ms", "wind_10m_ms", "temperature_c"):
        grouped = result[column].groupby(groups)
        additions[f"{column}_daily_anomaly"] = result[column] - grouped.transform("mean")
        additions[f"{column}_centered_mean_5"] = grouped.transform(
            lambda values: values.rolling(5, center=True, min_periods=1).mean()
        )
        additions[f"{column}_trend_6"] = grouped.shift(-3).fillna(result[column]) - grouped.shift(
            3
        ).fillna(result[column])
    if lead_days == 1:
        older_direction = np.deg2rad(weather.wind_direction_100m_previous_day2)
        additions.update(
            {
                "older_wind_10m": weather.wind_speed_10m_previous_day2 / 3.6,
                "older_temperature": weather.temperature_2m_previous_day2,
                "older_direction_sin": np.sin(older_direction),
                "older_direction_cos": np.cos(older_direction),
                "direction_agreement": np.cos(direction - older_direction),
                "temperature_run_change": result.temperature_c
                - weather.temperature_2m_previous_day2,
            }
        )
    if trajectory:
        # A daily forecast vector is available at the same nominal issue cutoff.
        for column in ("wind_100m_ms", "wind_u", "wind_v", "temperature_c"):
            for hour in range(0, 24, 3):
                selected = (
                    result[column].where(timestamp.dt.hour == hour).groupby(groups).transform("max")
                )
                additions[f"{column}_at_{hour:02d}"] = selected.fillna(result[column])
    return pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1)
