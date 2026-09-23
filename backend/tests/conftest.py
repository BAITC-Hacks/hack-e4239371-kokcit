import os

# Avoid excessive threads and OS-specific multiprocessing during model inference.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.services.weather import WEATHER_VARIABLES


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "database_path", tmp_path / "test.sqlite3")
    monkeypatch.setattr(settings, "weather_cache_dir", tmp_path / "weather")


@pytest.fixture
def february_weather():
    timestamps = pd.date_range("2026-02-01", "2026-02-28 23:00", freq="h")
    values = {"timestamp": timestamps}
    for lead in (1, 2):
        for variable in WEATHER_VARIABLES:
            values[f"{variable}_previous_day{lead}"] = (
                np.full(len(timestamps), 180.0)
                if "direction" in variable
                else np.full(len(timestamps), -2.0)
                if "temperature" in variable
                else 20 + 5 * np.sin(np.arange(len(timestamps)) / 8)
            )
    frame = pd.DataFrame(values)
    frame.attrs["source"] = "synthetic_test_fixture"
    return frame
