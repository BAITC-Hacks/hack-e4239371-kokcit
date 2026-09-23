import json
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd

from app.catalog import LOCAL_TIMEZONE, TURBINES
from app.config import settings
from app.schemas import ForecastPoint, ForecastRequest
from app.services.features import build_context_features, build_features
from app.services.model import load_model, model_digest
from app.services.weather import WEATHER_VARIABLES, fetch_archived_weather


def _model_error(turbine_id: int, lead_days: int) -> float:
    metrics_path = settings.model_dir / "metrics.json"
    if not metrics_path.exists():
        return 0.2
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(metrics[f"turbine_{turbine_id}"][f"day_{lead_days}"]["rmse"])


def validate_weather(request: ForecastRequest, weather: pd.DataFrame) -> list[pd.DataFrame]:
    if "timestamp" not in weather:
        raise ValueError("В погоде отсутствует временная шкала")
    timestamps = pd.to_datetime(weather["timestamp"], errors="raise")
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)
    weather = weather.copy()
    weather["timestamp"] = timestamps
    expected_hours = pd.date_range(
        str(request.issue_date + timedelta(days=1)), periods=request.horizon_hours, freq="h"
    )
    if not pd.DatetimeIndex(timestamps.sort_values()).equals(expected_hours):
        raise ValueError(
            "Ожидается ровно 24/48 последовательных часов: есть пропуски, дубли или лишние часы"
        )
    days = []
    for lead in range(1, request.horizon_hours // 24 + 1):
        valid_date = request.issue_date + timedelta(days=lead)
        day = weather[timestamps.dt.date == valid_date].copy()
        day["timestamp"] = pd.to_datetime(day.timestamp)
        day = day.sort_values("timestamp").reset_index(drop=True)
        expected = pd.date_range(str(valid_date), periods=24, freq="h")
        if len(day) != 24 or not pd.DatetimeIndex(day.timestamp).equals(expected):
            raise ValueError(f"Ожидаются 24 уникальных последовательных часа за {valid_date}")
        columns = [f"{variable}_previous_day{lead}" for variable in WEATHER_VARIABLES]
        if lead == 1:
            columns.append("wind_speed_100m_previous_day2")
        if not set(columns).issubset(day.columns):
            raise ValueError("В погоде отсутствуют обязательные поля")
        values = day[columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Пропуски или некорректные числа в погоде за {valid_date}")
        day[columns] = values
        for column in columns:
            if column.startswith("wind_speed") and (day[column] < 0).any():
                raise ValueError("Скорость ветра не может быть отрицательной")
            if column.startswith("wind_direction") and not day[column].between(0, 360).all():
                raise ValueError("Направление ветра должно быть в диапазоне 0–360 градусов")
        days.append(day)
    return days


def validate_issue_boundary(request: ForecastRequest, days: list[pd.DataFrame]) -> dict:
    """Check nominal forecast origins under the provider's fixed-lead contract.

    This proves an origin-time bound, not publication latency. The archive does
    not expose publication timestamps, so never label them as verified.
    """
    issue_at = datetime.combine(request.issue_date, time(23, 59, 59), LOCAL_TIMEZONE)
    origins = []
    for lead, day in enumerate(days, start=1):
        valid = pd.DatetimeIndex(day.timestamp).tz_localize(LOCAL_TIMEZONE)
        origins.extend(valid - pd.Timedelta(days=lead))
        if lead == 1:
            origins.extend(valid - pd.Timedelta(days=2))  # older-wind context feature
    if not origins or max(origins) > issue_at:
        raise ValueError("Прогноз погоды содержит номинальный выпуск позже момента расчёта")
    return {
        "archive_kind": "fixed_lead_offsets",
        "timezone": "Asia/Almaty",
        "issue_at": issue_at.isoformat(),
        "earliest_nominal_origin": min(origins).isoformat(),
        "latest_nominal_origin": max(origins).isoformat(),
        "nominal_origin_verified": True,
        "publication_time_verified": False,
        "single_initialization": False,
    }


def select_models(request: ForecastRequest) -> list[dict]:
    bundles = []
    for lead in range(1, request.horizon_hours // 24 + 1):
        model_path = settings.model_dir / f"turbine_{request.turbine_id}_day_{lead}.joblib"
        validation_path = settings.model_dir / "validation" / model_path.name
        if request.issue_date.isoformat() < "2026-01-31" and validation_path.exists():
            model_path = validation_path
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path.name}")
        bundle = load_model(model_path)
        if not isinstance(bundle, dict):
            bundle = {"estimator": bundle, "feature_mode": "basic", "trained_through": "2026-01-31"}
        if request.issue_date.isoformat() < bundle["trained_through"]:
            raise ValueError(
                "Дата выпуска раньше конца обучения модели; выберите более позднюю дату"
            )
        available_from = bundle.get(
            "available_from",
            "2025-12-31" if model_path == validation_path else bundle["trained_through"],
        )
        if request.issue_date.isoformat() < available_from:
            raise ValueError(
                "Модель ещё не была доступна на эту дату: выбор настроек и калибровка завершились позже"
            )
        bundles.append(
            {
                **bundle,
                "artifact": {
                    "lead_days": lead,
                    "version": bundle.get("model_version", "legacy"),
                    "trained_through": bundle["trained_through"],
                    "available_from": available_from,
                    "role": "validation" if model_path == validation_path else "production",
                    "sha256": model_digest(model_path),
                },
            }
        )
    return bundles


def prepare_inputs(
    request: ForecastRequest, days: list[pd.DataFrame], bundles: list[dict] | None = None
) -> list[tuple]:
    prepared = []
    bundles = select_models(request) if bundles is None else bundles
    for lead, (day, bundle) in enumerate(zip(days, bundles, strict=True), start=1):
        builder = build_context_features if bundle["feature_mode"] == "context" else build_features
        features = builder(day, lead)
        if not np.isfinite(features.to_numpy()).all():
            raise ValueError("Некорректные признаки модели")
        prepared.append((lead, day, features, bundle))
    return prepared


def _predict_day(request: ForecastRequest, prepared: tuple) -> list[ForecastPoint]:
    lead_days, day_weather, features, bundle = prepared
    raw_prediction = bundle["estimator"].predict(features)
    if not np.isfinite(raw_prediction).all():
        raise ValueError("Модель вернула некорректные значения")
    prediction = np.clip(raw_prediction, 0, 1)
    error = bundle.get("error_radius")
    if error is None:
        error = _model_error(request.turbine_id, lead_days)
    suffix = f"_previous_day{lead_days}"

    points = []
    for row_index, (_, row) in enumerate(day_weather.iterrows()):
        value = float(prediction[row_index])
        points.append(
            ForecastPoint(
                timestamp=row["timestamp"],
                normalized_power=value,
                confidence_low=max(0.0, value - error),
                confidence_high=min(1.0, value + error),
                wind_speed_100m_ms=float(row[f"wind_speed_100m{suffix}"] / 3.6),
                temperature_c=float(row[f"temperature_2m{suffix}"]),
            )
        )
    return points


def predict_prepared(request: ForecastRequest, prepared: list[tuple]) -> list[ForecastPoint]:
    return [point for day in prepared for point in _predict_day(request, day)]


def check_forecast(request: ForecastRequest, points: list[ForecastPoint]) -> list[str]:
    expected = pd.date_range(
        str(request.issue_date + timedelta(days=1)),
        periods=request.horizon_hours,
        freq="h",
        tz=LOCAL_TIMEZONE,
    )
    if not pd.DatetimeIndex([point.timestamp for point in points]).equals(expected):
        raise ValueError("Прогноз содержит пропуски или дубликаты часов")
    warnings = []
    clipped = sum(point.normalized_power in (0.0, 1.0) for point in points)
    if clipped > len(points) * 0.25:
        warnings.append("Более 25% значений находятся на физической границе мощности")
    return warnings


def calculate_forecast_from_weather(
    request: ForecastRequest, weather: pd.DataFrame
) -> tuple[list[ForecastPoint], list[str]]:
    prepared = prepare_inputs(request, validate_weather(request, weather))
    points = predict_prepared(request, prepared)
    return points, check_forecast(request, points)


def calculate_forecast(request: ForecastRequest) -> tuple[list[ForecastPoint], list[str]]:
    latitude, longitude = TURBINES[request.turbine_id]
    first_valid_day = request.issue_date + timedelta(days=1)
    final_valid_day = request.issue_date + timedelta(days=request.horizon_hours // 24)
    weather = fetch_archived_weather(
        latitude, longitude, first_valid_day, final_valid_day, request.data_mode
    )
    return calculate_forecast_from_weather(request, weather)
