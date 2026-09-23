from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_forecast_run_is_accepted() -> None:
    response = client.post(
        "/api/forecasts",
        json={"turbine_id": 1, "horizon_hours": 24, "issue_date": "2026-01-31"},
    )
    assert response.status_code == 202
    assert len(response.json()["steps"]) == 6
