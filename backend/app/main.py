from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent import accept_forecast_run, describe_pipeline
from app.config import settings
from app.schemas import AgentStep, ForecastAccepted, ForecastRequest

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


@app.post("/api/forecasts", response_model=ForecastAccepted, status_code=202)
def create_forecast(request: ForecastRequest) -> ForecastAccepted:
    return accept_forecast_run(request)
