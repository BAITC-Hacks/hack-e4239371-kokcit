from uuid import uuid4

from app.schemas import AgentStep, ForecastAccepted, ForecastRequest

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


def accept_forecast_run(request: ForecastRequest) -> ForecastAccepted:
    # The next milestone replaces this acceptance stub with the executable pipeline.
    _ = request
    return ForecastAccepted(run_id=str(uuid4()), steps=describe_pipeline())
