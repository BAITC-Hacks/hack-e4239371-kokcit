import hashlib
import json
import logging
import time
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from app.catalog import TURBINES
from app.config import settings

logger = logging.getLogger(__name__)

WEATHER_VARIABLES = (
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "temperature_2m",
)
TIMING_NOTE = (
    "Архив прогнозов с упреждением 24/48 часов; расчёт датирован концом дня "
    "(23:59:59 UTC+5). API не сообщает время публикации каждого выпуска. "
    "Будущие фактические измерения не используются."
)


class WeatherUnavailable(RuntimeError):
    pass


def _requested_variables() -> str:
    return ",".join(
        f"{variable}_previous_day{day}" for variable in WEATHER_VARIABLES for day in (1, 2)
    )


def validate_archive(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    expected = pd.date_range(start, f"{end} 23:00", freq="h")
    required = _requested_variables().split(",")
    if "timestamp" not in frame or not set(required).issubset(frame.columns):
        raise ValueError("В архиве погоды отсутствуют обязательные поля")
    if not pd.DatetimeIndex(frame.timestamp).equals(expected):
        raise ValueError("Архив погоды содержит пропуски, дубли или неожиданные часы")
    values = frame[required].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Архив погоды содержит некорректные числа")
    for column in required:
        if column.startswith("wind_speed") and (values[column] < 0).any():
            raise ValueError("Архив погоды содержит отрицательную скорость ветра")
        if column.startswith("wind_direction") and not values[column].between(0, 360).all():
            raise ValueError("Архив погоды содержит некорректное направление ветра")
    frame[required] = values
    return frame


def _read_snapshot(path: Path, start: str, end: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        wanted = pd.date_range(start, f"{end} 23:00", freq="h")
        frame = frame[frame.timestamp.isin(wanted)].sort_values("timestamp")
        return validate_archive(frame.reset_index(drop=True), start, end)
    except (ValueError, OSError, KeyError):
        return None


def fetch_archived_weather(
    latitude: float,
    longitude: float,
    start_date: date | str,
    end_date: date | str,
    mode: str = "auto",
    on_event=None,
) -> pd.DataFrame:
    if mode not in ("auto", "offline", "live"):
        raise ValueError("Неизвестный режим получения погоды")
    start, end = str(start_date), str(end_date)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "timezone": "Asia/Almaty",
        "wind_speed_unit": "kmh",
        "hourly": _requested_variables(),
    }
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:24]
    cache = settings.weather_cache_dir / f"{key}.csv"
    candidates = [cache]
    for turbine_id, coordinates in TURBINES.items():
        if coordinates == (latitude, longitude):
            candidates += [settings.demo_dir / f"weather_turbine_{turbine_id}.csv"]
    if mode != "live":
        for path in candidates:
            frame = _read_snapshot(path, start, end)
            if frame is not None:
                source = "bundled_archive" if path.parent == settings.demo_dir else "cache"
                frame.attrs.update(
                    source=source,
                    timing_note=TIMING_NOTE,
                    attempts=0,
                    archive_kind="fixed_lead_offsets",
                    publication_time_verified=False,
                )
                if on_event:
                    on_event(f"Загружен сохранённый архив: {len(frame)} часов")
                return frame
    if mode == "offline":
        raise WeatherUnavailable(
            "Для выбранных дат нет сохранённой погоды. Выберите демо 29 января 2026 "
            "или подготовьте архив командой python -m scripts.prepare_demo."
        )

    last_error = None
    for attempt in range(1, settings.weather_attempts + 1):
        if on_event:
            on_event(f"Запрос Open-Meteo, попытка {attempt}/{settings.weather_attempts}")
        try:
            response = httpx.get(
                settings.open_meteo_base_url,
                params=params,
                timeout=settings.weather_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                if len(payload) != 1:
                    raise ValueError("Ожидался ответ погоды для одной точки")
                payload = payload[0]
            if not isinstance(payload, dict) or not isinstance(payload.get("hourly"), dict):
                raise TypeError("Ответ Open-Meteo не содержит почасовой погоды")
            if not isinstance(payload["hourly"].get("time"), list):
                raise TypeError("Ответ Open-Meteo не содержит временной шкалы")
            if any(not isinstance(values, list) for values in payload["hourly"].values()):
                raise ValueError("Некорректный формат почасовой погоды")
            frame = pd.DataFrame(payload["hourly"])
            frame["timestamp"] = pd.to_datetime(frame.pop("time"), errors="raise")
            validate_archive(frame, start, end)
            frame.attrs.update(
                source="open_meteo",
                timing_note=TIMING_NOTE,
                attempts=attempt,
                archive_kind="fixed_lead_offsets",
                publication_time_verified=False,
            )
            try:
                settings.weather_cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(f".{time.time_ns()}.tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(cache)
            except OSError:
                logger.warning("Weather cache is not writable; using validated network data")
                frame.attrs["cache_warning"] = (
                    "Погода получена, но локальный кэш недоступен для записи"
                )
                if on_event:
                    on_event(frame.attrs["cache_warning"])
            return frame
        except httpx.HTTPStatusError as error:
            last_error = error
            if error.response.status_code < 500 and error.response.status_code != 429:
                break
        except (httpx.RequestError, ValueError, KeyError, TypeError) as error:
            last_error = error
        if attempt < settings.weather_attempts:
            time.sleep(min(attempt * 0.3, 1))
    raise WeatherUnavailable(
        "Не удалось получить прогноз погоды. Повторите позже или выберите "
        "сохранённый архив (демо: 29 января 2026)."
    ) from last_error
