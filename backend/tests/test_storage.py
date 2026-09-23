from datetime import UTC, date, datetime

import pytest

from app.config import settings
from app.schemas import AgentStep, ForecastAccepted, ForecastPoint
from app.services.storage import (
    february_forecast_to_csv,
    get_forecast,
    list_forecasts,
    save_forecast,
)


def make_result(run_id: str, power: float) -> ForecastAccepted:
    return ForecastAccepted(
        run_id=run_id,
        turbine_id=1,
        issue_date=date(2026, 1, 31),
        horizon_hours=24,
        expected_normalized_energy=power,
        steps=[AgentStep(id="done", title="Done", status="completed")],
        forecast=[
            ForecastPoint(
                timestamp=datetime(2026, 2, 1, tzinfo=UTC),
                normalized_power=power,
                confidence_low=max(0, power - 0.1),
                confidence_high=min(1, power + 0.1),
                wind_speed_100m_ms=7.0,
                temperature_c=-2.0,
            )
        ],
    )


def test_storage_replaces_same_logical_forecast(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "database_path", tmp_path / "test.sqlite3")
    save_forecast(make_result("first", 0.4))
    save_forecast(make_result("second", 0.6))

    history = list_forecasts()
    assert len(history) == 1
    assert history[0].run_id == "second"
    assert get_forecast("first") is None
    assert get_forecast("second").forecast[0].normalized_power == 0.6

    with pytest.raises(ValueError):
        february_forecast_to_csv(1)
