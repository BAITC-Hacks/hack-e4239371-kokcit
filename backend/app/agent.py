from uuid import uuid4

from app.schemas import AgentStep, ForecastAccepted, ForecastRequest
from app.services.forecast import calculate_forecast, calculate_forecast_from_weather

PIPELINE_STEPS = (
    ("weather", "Получение архивного прогноза погоды"),
    ("validation", "Проверка входных данных"),
    ("features", "Подготовка признаков"),
    ("forecast", "Расчет выработки"),
    ("quality", "Анализ и проверка результата"),
    ("persist", "Сохранение прогноза"),
)


def describe_pipeline() -> list[AgentStep]:
    return [AgentStep(id=step_id, title=title) for step_id, title in PIPELINE_STEPS]


def _build_result(request: ForecastRequest, points, warnings: list[str]) -> ForecastAccepted:
    steps = describe_pipeline()
    for step in steps:
        step.status = "completed"

    return ForecastAccepted(
        run_id=str(uuid4()),
        turbine_id=request.turbine_id,
        issue_date=request.issue_date,
        horizon_hours=request.horizon_hours,
        expected_normalized_energy=sum(point.normalized_power for point in points),
        quality_warnings=warnings,
        steps=steps,
        forecast=points,
    )


def run_forecast(request: ForecastRequest) -> ForecastAccepted:
    points, warnings = calculate_forecast(request)
    return _build_result(request, points, warnings)


def run_forecast_with_weather(request: ForecastRequest, weather) -> ForecastAccepted:
    points, warnings = calculate_forecast_from_weather(request, weather)
    return _build_result(request, points, warnings)
