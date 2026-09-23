import json
from datetime import date, timedelta

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.agent import describe_pipeline, run_forecast, run_forecast_with_weather
from app.config import settings
from app.schemas import (
    AgentStep,
    FebruaryForecastResponse,
    ForecastAccepted,
    ForecastRequest,
    ForecastSummary,
)
from app.services.storage import (
    february_forecast_to_csv,
    forecast_to_csv,
    get_forecast,
    list_forecasts,
    save_forecast,
)
from app.services.weather import fetch_archived_weather

TURBINES = {
    1: (43.645150, 78.535604),
    2: (43.643198, 78.538828),
}

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.get("/api/agent/steps", response_model=list[AgentStep])
def agent_steps() -> list[AgentStep]:
    return describe_pipeline()


@app.get("/api/models/metrics")
def model_metrics() -> dict:
    metrics_path = settings.model_dir / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Model metrics are not available")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.post("/api/forecasts", response_model=ForecastAccepted)
def create_forecast(request: ForecastRequest) -> ForecastAccepted:
    try:
        result = run_forecast(request)
        save_forecast(result)
        return result
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/forecasts/february", response_model=FebruaryForecastResponse)
def create_february_forecast(
    turbine_id: int = Query(ge=1, le=2),
) -> FebruaryForecastResponse:
    first_issue = date(2026, 1, 31)
    last_issue = date(2026, 2, 27)
    latitude, longitude = TURBINES[turbine_id]
    try:
        weather = fetch_archived_weather(latitude, longitude, "2026-02-01", "2026-02-28")
        current_issue = first_issue
        results = []
        while current_issue <= last_issue:
            request = ForecastRequest(
                turbine_id=turbine_id,
                issue_date=current_issue,
                horizon_hours=24,
            )
            result = run_forecast_with_weather(request, weather)
            save_forecast(result)
            results.append(result)
            current_issue += timedelta(days=1)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return FebruaryForecastResponse(
        turbine_id=turbine_id,
        generated_runs=len(results),
        first_issue_date=first_issue,
        last_issue_date=last_issue,
        expected_normalized_energy=sum(result.expected_normalized_energy for result in results),
        steps=results[-1].steps,
        forecast=[point for result in results for point in result.forecast],
    )


@app.get("/api/forecasts/february/export.csv")
def export_february_forecast(turbine_id: int = Query(ge=1, le=2)) -> Response:
    content = february_forecast_to_csv(turbine_id)
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
