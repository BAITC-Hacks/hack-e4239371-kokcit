import json
from datetime import timedelta

import numpy as np
import pandas as pd

from app.config import settings
from app.schemas import ForecastPoint, ForecastRequest
from app.services.features import build_features
from app.services.model import load_model
from app.services.weather import fetch_archived_weather

TURBINES = {
    1: (43.645150, 78.535604),
    2: (43.643198, 78.538828),
}


def _model_error(turbine_id: int, lead_days: int) -> float:
    metrics_path = settings.model_dir / "metrics.json"
    if not metrics_path.exists():
        return 0.2
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(metrics[f"turbine_{turbine_id}"][f"day_{lead_days}"]["rmse"])


def _forecast_day(
    request: ForecastRequest,
    weather: pd.DataFrame,
    lead_days: int,
) -> list[ForecastPoint]:
    valid_date = request.issue_date + timedelta(days=lead_days)
    day_weather = weather[pd.to_datetime(weather["timestamp"]).dt.date == valid_date].copy()
    if len(day_weather) != 24:
        raise ValueError(f"Expected 24 weather hours for {valid_date}, found {len(day_weather)}")

    features = build_features(day_weather, lead_days)
    model_path = settings.model_dir / f"turbine_{request.turbine_id}_day_{lead_days}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run the training script first.")

    model = load_model(model_path)
    prediction = np.clip(model.predict(features), 0, 1)
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


def calculate_forecast_from_weather(
    request: ForecastRequest, weather: pd.DataFrame
) -> tuple[list[ForecastPoint], list[str]]:
    points = _forecast_day(request, weather, 1)
    if request.horizon_hours == 48:
        points.extend(_forecast_day(request, weather, 2))

    warnings = []
    clipped = sum(point.normalized_power in (0.0, 1.0) for point in points)
    if clipped > len(points) * 0.25:
        warnings.append("Более 25% значений находятся на физической границе мощности")
    return points, warnings


def calculate_forecast(request: ForecastRequest) -> tuple[list[ForecastPoint], list[str]]:
    latitude, longitude = TURBINES[request.turbine_id]
    first_valid_day = request.issue_date + timedelta(days=1)
    final_valid_day = request.issue_date + timedelta(days=request.horizon_hours // 24)
    weather = fetch_archived_weather(latitude, longitude, first_valid_day, final_valid_day)
    return calculate_forecast_from_weather(request, weather)
