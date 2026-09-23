from pathlib import Path

import pandas as pd

COLUMNS = ("id", "timestamp", "measured_wind_ms", "power", "measured_temperature_c")


def load_turbine_csv(path: str | Path, turbine_id: int) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if len(frame.columns) != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} columns, found {len(frame.columns)}")

    frame.columns = COLUMNS
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")

    hourly = (
        frame.set_index("timestamp")
        .resample("1h")[["measured_wind_ms", "power", "measured_temperature_c"]]
        .mean()
        .dropna(subset=["power"])
        .reset_index()
    )
    hourly["turbine_id"] = turbine_id
    return hourly
