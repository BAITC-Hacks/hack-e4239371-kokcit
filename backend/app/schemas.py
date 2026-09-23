from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.catalog import LOCAL_TIMEZONE


class AgentStep(BaseModel):
    id: str
    title: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = None
    detail: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ForecastRequest(BaseModel):
    turbine_id: Literal[1, 2]
    horizon_hours: Literal[24, 48] = 24
    issue_date: date = Field(description="Дата выпуска прогноза")
    data_mode: Literal["auto", "offline", "live"] = "auto"


class ForecastPoint(BaseModel):
    timestamp: datetime
    normalized_power: float = Field(ge=0, le=1)
    confidence_low: float = Field(ge=0, le=1)
    confidence_high: float = Field(ge=0, le=1)
    wind_speed_100m_ms: float = Field(ge=0, allow_inf_nan=False)
    temperature_c: float = Field(allow_inf_nan=False)

    @field_validator("timestamp")
    @classmethod
    def local_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=LOCAL_TIMEZONE)
        return value.astimezone(LOCAL_TIMEZONE)


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
    data_source: str = "unknown"
    model_version: str = "legacy"
    issue_time: str = "23:59:59+05:00"
    weather_run: str | None = None
    timing_note: str = ""
    issue_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    weather_provenance: dict[str, Any] = Field(default_factory=dict)
    model_artifacts: list[dict[str, Any]] = Field(default_factory=list)


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
    quality_warnings: list[str] = Field(default_factory=list)
    data_source: str = "unknown"
    model_version: str = "legacy"
    timing_note: str = ""
    publication_time_verified: bool = False


class ForecastJob(BaseModel):
    job_id: str
    kind: Literal["forecast", "february"] = "forecast"
    request: ForecastRequest | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    steps: list[AgentStep] = Field(default_factory=list)
    progress: int = 0
    total: int = 1
    message: str = "Ожидание запуска"
    error: str | None = None
    result: ForecastAccepted | FebruaryForecastResponse | None = None
