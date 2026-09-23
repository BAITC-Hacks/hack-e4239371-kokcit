import time

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.agent import run_february
from app.config import settings
from app.main import app
from app.schemas import ForecastRequest
from app.services.forecast import select_models, validate_issue_boundary, validate_weather
from app.services.weather import fetch_archived_weather

client = TestClient(app)


@pytest.mark.parametrize(
    "payload", [None, [], [None], {}, {"hourly": None}, {"hourly": {"time": 3}}]
)
def test_malformed_weather_is_a_safe_service_error(monkeypatch, payload):
    calls = []

    def get(*_args, **_kwargs):
        calls.append(1)
        return httpx.Response(
            200, json=payload, request=httpx.Request("GET", "https://example.test")
        )

    monkeypatch.setattr(settings, "weather_attempts", 2)
    monkeypatch.setattr("app.services.weather.httpx.get", get)
    monkeypatch.setattr("app.services.weather.time.sleep", lambda _: None)
    response = client.post(
        "/api/forecasts",
        json={
            "turbine_id": 1,
            "issue_date": "2026-01-31",
            "data_mode": "live",
        },
    )
    assert response.status_code == 503
    assert len(calls) == 2
    assert "Traceback" not in response.text
    assert client.get("/api/forecasts").json() == []


def test_transient_failure_recovers_and_verified_response_is_cached(monkeypatch, february_weather):
    frame = february_weather.iloc[:24].copy()
    payload = frame.drop(columns="timestamp").to_dict(orient="list")
    payload["time"] = frame.timestamp.dt.strftime("%Y-%m-%dT%H:%M").tolist()
    calls = []

    def get(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("temporary")
        return httpx.Response(
            200, json={"hourly": payload}, request=httpx.Request("GET", "https://example.test")
        )

    monkeypatch.setattr("app.services.weather.httpx.get", get)
    monkeypatch.setattr("app.services.weather.time.sleep", lambda _: None)
    first = fetch_archived_weather(43.0, 78.0, "2026-02-01", "2026-02-01", mode="live")
    assert first.attrs["source"] == "open_meteo"
    assert first.attrs["attempts"] == 2
    cached = fetch_archived_weather(43.0, 78.0, "2026-02-01", "2026-02-01")
    assert cached.attrs["source"] == "cache"
    assert len(calls) == 2
    pd.testing.assert_frame_equal(first, cached, check_freq=False)

    # Corrupt cache must never become a successful forecast or prevent recovery.
    cache_path = next(settings.weather_cache_dir.glob("*.csv"))
    broken = cached.copy()
    broken.loc[0, "wind_speed_100m_previous_day1"] = -1
    broken.to_csv(cache_path, index=False)
    recovered = fetch_archived_weather(43.0, 78.0, "2026-02-01", "2026-02-01")
    assert recovered.attrs["source"] == "open_meteo"
    assert len(calls) == 3


def test_extra_hours_are_rejected(february_weather):
    with pytest.raises(ValueError, match="лишние часы"):
        validate_weather(
            ForecastRequest(turbine_id=1, issue_date="2026-01-31"), february_weather.iloc[:25]
        )


def test_nominal_origins_are_checked_without_claiming_publication_timestamps(february_weather):
    request = ForecastRequest(turbine_id=1, issue_date="2026-01-31", horizon_hours=48)
    days = validate_weather(request, february_weather.iloc[:48])
    provenance = validate_issue_boundary(request, days)
    assert provenance["issue_at"] == "2026-01-31T23:59:59+05:00"
    assert provenance["latest_nominal_origin"] == "2026-01-31T23:00:00+05:00"
    assert provenance["nominal_origin_verified"] is True
    assert provenance["publication_time_verified"] is False
    days[1]["timestamp"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="позже"):
        validate_issue_boundary(request, days)


def test_models_cannot_be_used_before_parameter_selection_finished(monkeypatch):
    monkeypatch.setattr(
        "app.services.forecast.load_model",
        lambda _path: {
            "trained_through": "2025-12-01",
            "available_from": "2025-12-31",
            "feature_mode": "basic",
        },
    )
    with pytest.raises(ValueError, match="выбор настроек"):
        select_models(ForecastRequest(turbine_id=1, issue_date="2025-12-30", horizon_hours=48))


def test_month_validation_failure_is_recorded_before_any_writes(monkeypatch, february_weather):
    broken = february_weather.copy()
    broken.loc[671, "timestamp"] = broken.loc[670, "timestamp"]
    monkeypatch.setattr("app.agent.fetch_archived_weather", lambda *_a, **_k: broken)
    traces = []
    with pytest.raises(ValueError, match="672"):
        run_february(
            1, on_update=lambda steps: traces.append([step.model_dump() for step in steps])
        )
    assert next(step for step in traces[-1] if step["id"] == "validation")["status"] == "failed"
    assert client.get("/api/forecasts").json() == []


def test_failed_job_trace_survives_memory_registry_reset():
    response = client.post(
        "/api/jobs",
        json={
            "turbine_id": 1,
            "issue_date": "2027-01-01",
            "data_mode": "offline",
        },
    )
    job_id = response.json()["job_id"]
    for _ in range(200):
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["status"] == "failed":
            break
        time.sleep(0.01)
    assert result["status"] == "failed"
    with jobs._lock:
        jobs._jobs.pop(job_id)
    restored = client.get(f"/api/jobs/{job_id}").json()
    assert restored == result
    history = client.get("/api/jobs").json()
    assert history[0]["job_id"] == job_id
    assert history[0]["steps"][0]["status"] == "failed"
    assert "result" not in history[0]


def test_february_job_records_effective_request_and_completes(monkeypatch, february_weather):
    monkeypatch.setattr("app.agent.fetch_archived_weather", lambda *_a, **_k: february_weather)
    response = client.post(
        "/api/jobs/february",
        json={
            "turbine_id": 1,
            "issue_date": "2026-01-29",
            "horizon_hours": 48,
            "data_mode": "offline",
        },
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    for _ in range(500):
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["status"] in ("completed", "failed"):
            break
        time.sleep(0.01)
    assert result["status"] == "completed", result
    assert result["kind"] == "february"
    assert result["request"]["issue_date"] == "2026-01-31"
    assert result["request"]["horizon_hours"] == 24
    assert result["progress"] == result["total"] == 28
    assert len(result["result"]["forecast"]) == 672
