from typing import Literal

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    id: str
    title: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"


class ForecastRequest(BaseModel):
    turbine_id: Literal[1, 2]
    horizon_hours: Literal[24, 48] = 24
    issue_date: str = Field(description="Дата выпуска прогноза в формате YYYY-MM-DD")


class ForecastAccepted(BaseModel):
    run_id: str
    status: Literal["accepted"] = "accepted"
    steps: list[AgentStep]
