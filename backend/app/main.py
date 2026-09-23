import json
import logging
import sqlite3
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.agent import describe_pipeline, run_february, run_forecast
from app.config import settings
from app.jobs import get_job, submit
from app.schemas import (
    AgentStep,
    FebruaryForecastResponse,
    ForecastAccepted,
    ForecastJob,
    ForecastRequest,
    ForecastSummary,
)
from app.services.storage import (
    february_forecast_to_csv,
    forecast_to_csv,
    get_forecast,
    list_forecasts,
    list_job_audits,
)
from app.services.weather import WeatherUnavailable

app = FastAPI(title=settings.app_name, version="0.1.0")
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(WeatherUnavailable)
async def weather_error(_request: Request, error: WeatherUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(sqlite3.Error)
@app.exception_handler(OSError)
async def storage_error(_request: Request, error: Exception):
    logger.error("Local storage unavailable: %s", type(error).__name__)
    return JSONResponse(
        status_code=503,
        content={"detail": "Локальное хранилище недоступно. Проверьте доступ к папке данных."},
    )


@app.get("/api/health")
def health() -> dict:
    ready = all(
        (settings.model_dir / f"turbine_{t}_day_{d}.joblib").exists()
        for t in (1, 2)
        for d in (1, 2)
    )
    return {"status": "ok", "service": settings.app_name, "models_ready": ready}


@app.get("/api/agent/steps", response_model=list[AgentStep])
def agent_steps() -> list[AgentStep]:
    return describe_pipeline()


@app.get("/api/models/metrics")
def model_metrics() -> dict:
    metrics_path = settings.model_dir / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Model metrics are not available")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.get("/api/models/validation")
def model_validation(turbine_id: int = Query(ge=1, le=2), lead_days: int = Query(ge=1, le=2)):
    path = settings.model_dir / "validation_predictions.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Отчёт проверки пока не подготовлен")
    frame = pd.read_csv(path)
    selected = frame[(frame.turbine_id == turbine_id) & (frame.lead_days == lead_days)]
    return {"points": selected.to_dict(orient="records"), "period": "Январь 2026"}


@app.post("/api/jobs", response_model=ForecastJob, status_code=202)
def create_job(request: ForecastRequest):
    try:
        return submit(request)
    except ValueError as error:
        raise HTTPException(status_code=429, detail=str(error)) from error


@app.post("/api/jobs/february", response_model=ForecastJob, status_code=202)
def create_february_job(request: ForecastRequest):
    try:
        return submit(request, february=True)
    except ValueError as error:
        raise HTTPException(status_code=429, detail=str(error)) from error


@app.get("/api/jobs")
def job_history(limit: int = Query(default=20, ge=1, le=100)):
    return list_job_audits(limit)


@app.get("/api/jobs/{job_id}", response_model=ForecastJob)
def job_status(job_id: str):
    result = get_job(job_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail="Расчёт не найден. После перезапуска сервера откройте историю."
        )
    return result


@app.post("/api/forecasts", response_model=ForecastAccepted)
def create_forecast(request: ForecastRequest) -> ForecastAccepted:
    try:
        result = run_forecast(request)
        return result
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/forecasts/february", response_model=FebruaryForecastResponse)
def create_february_forecast(
    turbine_id: int = Query(ge=1, le=2),
    data_mode: Literal["auto", "offline", "live"] = "auto",
) -> FebruaryForecastResponse:
    try:
        return run_february(turbine_id, data_mode)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/forecasts/february/export.csv")
def export_february_forecast(turbine_id: int = Query(ge=1, le=2)) -> Response:
    try:
        content = february_forecast_to_csv(turbine_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if content is None:
        raise HTTPException(status_code=404, detail="February forecast not found")
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="forecast-february-2026-turbine-{turbine_id}.csv"'
            )
        },
    )


@app.get("/api/forecasts", response_model=list[ForecastSummary])
def forecast_history(limit: int = Query(default=20, ge=1, le=100)) -> list[ForecastSummary]:
    return list_forecasts(limit)


@app.get("/api/forecasts/{run_id}", response_model=ForecastAccepted)
def forecast_details(run_id: str) -> ForecastAccepted:
    result = get_forecast(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Forecast run not found")
    return result


@app.get("/api/forecasts/{run_id}/export.csv")
def export_forecast(run_id: str) -> Response:
    result = get_forecast(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Forecast run not found")
    return Response(
        content=forecast_to_csv(result),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="forecast-{run_id}.csv"'},
    )


if settings.frontend_dir.exists():
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
