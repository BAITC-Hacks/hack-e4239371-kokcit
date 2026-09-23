from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from time import perf_counter
from uuid import uuid4

import pandas as pd

from app.catalog import MODEL_VERSION, TURBINES
from app.schemas import AgentStep, FebruaryForecastResponse, ForecastAccepted, ForecastRequest
from app.services.forecast import (
    check_forecast,
    predict_prepared,
    prepare_inputs,
    select_models,
    validate_issue_boundary,
    validate_weather,
)
from app.services.storage import save_forecast, update_forecast_trace
from app.services.weather import TIMING_NOTE, fetch_archived_weather

PIPELINE_STEPS = (
    ("weather", "Получение архивного прогноза погоды"),
    ("validation", "Проверка входных данных"),
    ("model_selection", "Выбор модели по турбине, горизонту и дате"),
    ("features", "Подготовка признаков"),
    ("forecast", "Расчет выработки"),
    ("quality", "Анализ и проверка результата"),
    ("persist", "Сохранение прогноза"),
)


def describe_pipeline() -> list[AgentStep]:
    return [AgentStep(id=step_id, title=title) for step_id, title in PIPELINE_STEPS]


class PipelineTrace:
    def __init__(self, on_update=None):
        self.steps = describe_pipeline()
        self.on_update = on_update

    def publish(self):
        if self.on_update:
            self.on_update(self.steps)

    def detail(self, step_id, text):
        step = next(step for step in self.steps if step.id == step_id)
        step.detail = text
        step.metadata.setdefault("events", []).append(
            {"at": datetime.now(UTC).isoformat(), "message": text}
        )
        self.publish()

    @contextmanager
    def stage(self, step_id):
        step = next(step for step in self.steps if step.id == step_id)
        step.status, step.started_at = "running", datetime.now(UTC)
        start = perf_counter()
        self.publish()
        try:
            yield step
        except Exception as error:
            from app.services.weather import WeatherUnavailable

            step.status = "failed"
            step.detail = (
                str(error)
                if isinstance(error, (ValueError, FileNotFoundError, WeatherUnavailable))
                else "Внутренняя ошибка этапа. Проверьте журнал сервера."
            )
            raise
        else:
            step.status = "completed"
        finally:
            step.finished_at = datetime.now(UTC)
            step.duration_ms = round((perf_counter() - start) * 1000, 2)
            self.publish()


def run_forecast(request: ForecastRequest, on_update=None, weather=None) -> ForecastAccepted:
    trace = PipelineTrace(on_update)
    with trace.stage("weather") as step:
        if weather is None:
            latitude, longitude = TURBINES[request.turbine_id]
            weather = fetch_archived_weather(
                latitude,
                longitude,
                request.issue_date + timedelta(days=1),
                request.issue_date + timedelta(days=request.horizon_hours // 24),
                mode=request.data_mode,
                on_event=lambda text: trace.detail("weather", text),
            )
        step.detail = f"Источник: {weather.attrs.get('source', 'provided')}; {len(weather)} часов"
        step.metadata.update(
            source=weather.attrs.get("source", "provided"),
            attempts=weather.attrs.get("attempts", 0),
        )
    with trace.stage("validation") as step:
        days = validate_weather(request, weather)
        provenance = validate_issue_boundary(request, days)
        step.metadata.update(provenance)
        step.detail = f"{request.horizon_hours} уникальных часов, обязательные поля без пропусков"
    with trace.stage("model_selection") as step:
        bundles = select_models(request)
        step.metadata["models"] = [bundle["artifact"] for bundle in bundles]
        step.detail = "Модели выбраны; конец обучения не позже даты выпуска прогноза"
    with trace.stage("features") as step:
        prepared = prepare_inputs(request, days, bundles)
        step.metadata["feature_counts"] = [len(item[2].columns) for item in prepared]
        step.detail = f"{len(prepared[0][2].columns)} признаков; только прогноз погоды и календарь"
    with trace.stage("forecast") as step:
        points = predict_prepared(request, prepared)
        step.detail = f"Турбина {request.turbine_id}; рассчитано {len(points)} значений"
    with trace.stage("quality") as step:
        warnings = check_forecast(request, points)
        warnings.append(
            "Время публикации погодных выпусков не подтверждено архивом; проверены номинальные упреждения 24/48 часов"
        )
        if weather.attrs.get("cache_warning"):
            warnings.append(weather.attrs["cache_warning"])
        step.detail = "; ".join(warnings) if warnings else "Полная временная сетка, мощность 0–1"
    result = ForecastAccepted(
        run_id=str(uuid4()),
        turbine_id=request.turbine_id,
        issue_date=request.issue_date,
        horizon_hours=request.horizon_hours,
        expected_normalized_energy=sum(point.normalized_power for point in points),
        quality_warnings=warnings,
        steps=trace.steps,
        forecast=points,
        data_source=weather.attrs.get("source", "provided"),
        model_version=MODEL_VERSION,
        timing_note=weather.attrs.get("timing_note", TIMING_NOTE),
        issue_at=provenance["issue_at"],
        weather_provenance={**provenance, "source": weather.attrs.get("source", "provided")},
        model_artifacts=[bundle["artifact"] for bundle in bundles],
    )
    with trace.stage("persist") as step:
        save_forecast(result)
        step.detail = "Прогноз и метаданные сохранены в SQLite"
    update_forecast_trace(result)
    return result


def run_forecast_with_weather(request: ForecastRequest, weather) -> ForecastAccepted:
    return run_forecast(request, weather=weather)


def run_february(turbine_id: int, data_mode: str = "auto", on_update=None, on_progress=None):
    first_issue, last_issue = date(2026, 1, 31), date(2026, 2, 27)
    first_request = ForecastRequest(
        turbine_id=turbine_id, issue_date=first_issue, data_mode=data_mode
    )
    trace = PipelineTrace(on_update)
    with trace.stage("weather"):
        weather = fetch_archived_weather(
            *TURBINES[turbine_id],
            "2026-02-01",
            "2026-02-28",
            mode=data_mode,
            on_event=lambda text: trace.detail("weather", text),
        )
    # Validate the entire month before persisting any part of this batch.
    with trace.stage("validation") as step:
        expected = pd.date_range("2026-02-01", "2026-02-28 23:00", freq="h")
        if "timestamp" not in weather or not pd.DatetimeIndex(weather.timestamp).equals(expected):
            raise ValueError("Февраль должен содержать ровно 672 последовательных часа")
        for offset in range(28):
            daily_request = first_request.model_copy(
                update={"issue_date": first_issue + timedelta(days=offset)}
            )
            days = validate_weather(daily_request, weather.iloc[offset * 24 : (offset + 1) * 24])
            validate_issue_boundary(daily_request, days)
        step.detail = "Проверены 28 дней и 672 часа до начала расчётов"
    results = []
    for offset in range(28):
        request = first_request.model_copy(
            update={"issue_date": first_issue + timedelta(days=offset)}
        )
        results.append(
            run_forecast(
                request, on_update=on_update, weather=weather.iloc[offset * 24 : (offset + 1) * 24]
            )
        )
        if on_progress:
            on_progress(offset + 1, 28)
    return FebruaryForecastResponse(
        turbine_id=turbine_id,
        generated_runs=len(results),
        first_issue_date=first_issue,
        last_issue_date=last_issue,
        expected_normalized_energy=sum(result.expected_normalized_energy for result in results),
        steps=results[-1].steps,
        forecast=[point for result in results for point in result.forecast],
        quality_warnings=sorted(
            {warning for result in results for warning in result.quality_warnings}
        ),
        data_source=results[-1].data_source,
        model_version=MODEL_VERSION,
        timing_note=TIMING_NOTE,
        publication_time_verified=False,
    )
