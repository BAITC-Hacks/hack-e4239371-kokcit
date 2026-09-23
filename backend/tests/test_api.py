from fastapi.testclient import TestClient

from app.main import app
from app.schemas import AgentStep, ForecastAccepted, ForecastPoint

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_forecast_run_is_completed(monkeypatch) -> None:
    def fake_forecast(_request):
        return (
            [
                ForecastPoint(
                    timestamp="2026-02-01T00:00:00",
                    normalized_power=0.5,
                    confidence_low=0.3,
                    confidence_high=0.7,
                    wind_speed_100m_ms=7.2,
                    temperature_c=-1.0,
                )
            ],
            [],
        )

    monkeypatch.setattr("app.agent.calculate_forecast", fake_forecast)
    monkeypatch.setattr("app.main.save_forecast", lambda _result: None)
    response = client.post(
        "/api/forecasts",
        json={"turbine_id": 1, "horizon_hours": 24, "issue_date": "2026-01-31"},
    )
    assert response.status_code == 200
    assert len(response.json()["steps"]) == 6
    assert response.json()["status"] == "completed"
    assert response.json()["forecast"][0]["normalized_power"] == 0.5


def test_february_run_returns_all_672_hours(monkeypatch) -> None:
    def fake_batch_forecast(request, _weather):
        points = [
            ForecastPoint(
                timestamp=f"{(request.issue_date.replace(day=1))}T{hour:02d}:00:00",
                normalized_power=0.5,
                confidence_low=0.3,
                confidence_high=0.7,
                wind_speed_100m_ms=7.2,
                temperature_c=-1.0,
            )
            for hour in range(24)
        ]
        return ForecastAccepted(
            run_id=f"run-{request.issue_date}",
            turbine_id=request.turbine_id,
            issue_date=request.issue_date,
            horizon_hours=24,
            expected_normalized_energy=12.0,
            steps=[AgentStep(id="done", title="Done", status="completed")],
            forecast=points,
        )

    monkeypatch.setattr("app.main.fetch_archived_weather", lambda *_args: object())
    monkeypatch.setattr("app.main.run_forecast_with_weather", fake_batch_forecast)
    monkeypatch.setattr("app.main.save_forecast", lambda _result: None)

    response = client.post("/api/forecasts/february?turbine_id=1")
    assert response.status_code == 200
    assert response.json()["generated_runs"] == 28
    assert len(response.json()["forecast"]) == 672
