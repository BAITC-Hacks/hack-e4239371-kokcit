from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    id: str
    title: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"


class ForecastRequest(BaseModel):
    turbine_id: Literal[1, 2]
    horizon_hours: Literal[24, 48] = 24
    issue_date: date = Field(description="Дата выпуска прогноза")


class ForecastPoint(BaseModel):
    timestamp: datetime
    normalized_power: float = Field(ge=0, le=1)
    confidence_low: float = Field(ge=0, le=1)
    confidence_high: float = Field(ge=0, le=1)
    wind_speed_100m_ms: float
    temperature_c: float


class ForecastAccepted(BaseModel):
    run_id: str
    status: Literal["completed"] = "completed"
    turbine_id: Literal[1, 2]
    issue_date: date
    horizon_hours: Literal[24, 48]
    expected_normalized_energy: float
    quality_warnings: list[str] = Field(default_factory=list)
    steps: list[AgentStep]
    forecast: list[ForecastPoint]


class ForecastSummary(BaseModel):
    run_id: str
    turbine_id: Literal[1, 2]
    issue_date: date
    horizon_hours: Literal[24, 48]
    expected_normalized_energy: float
    created_at: datetime


class FebruaryForecastResponse(BaseModel):
    turbine_id: Literal[1, 2]
    generated_runs: int
    first_issue_date: date
    last_issue_date: date
    expected_normalized_energy: float
    steps: list[AgentStep]
    forecast: list[ForecastPoint]
