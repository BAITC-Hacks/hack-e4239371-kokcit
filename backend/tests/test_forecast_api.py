import csv
import io
import json
import time

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import mean_absolute_error, r2_score

from app.config import settings
from app.main import app
from app.schemas import ForecastRequest
from app.services.features import build_context_features
from app.services.forecast import validate_weather
from app.services.model import load_model

client = TestClient(app)


def request_body(turbine=1, horizon=48):
    return {
        "turbine_id": turbine,
        "issue_date": "2026-01-29",
        "horizon_hours": horizon,
        "data_mode": "offline",
    }


def test_health_and_validation_report():
    assert client.get("/api/health").json()["models_ready"]
    response = client.get("/api/models/validation?turbine_id=1&lead_days=1")
    assert response.status_code == 200
    assert len(response.json()["points"]) == 744


@pytest.mark.parametrize("turbine", [1, 2])
@pytest.mark.parametrize("horizon", [24, 48])
def test_real_offline_model_pipeline_persistence_and_csv(turbine, horizon):
    response = client.post("/api/forecasts", json=request_body(turbine, horizon))
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["forecast"]) == horizon
    assert len({point["timestamp"] for point in data["forecast"]}) == horizon
    assert all(point["timestamp"].endswith("+05:00") for point in data["forecast"])
    assert all(
        0 <= point["confidence_low"] <= point["normalized_power"] <= point["confidence_high"] <= 1
        for point in data["forecast"]
    )
    assert data["data_source"] == "bundled_archive"
    assert all(step["status"] == "completed" and step["duration_ms"] >= 0 for step in data["steps"])
    stored = client.get(f"/api/forecasts/{data['run_id']}").json()
    assert stored["steps"] == data["steps"]
    assert stored["model_version"] == data["model_version"]
    assert stored["weather_provenance"] == data["weather_provenance"]
    assert stored["model_artifacts"] == data["model_artifacts"]
    assert stored["created_at"] == data["created_at"]
    assert all(len(artifact["sha256"]) == 64 for artifact in data["model_artifacts"])
    exported = client.get(f"/api/forecasts/{data['run_id']}/export.csv")
    assert len(list(csv.DictReader(io.StringIO(exported.text)))) == horizon
    report = pd.read_csv(settings.model_dir / "validation_predictions.csv")
    for index, point in enumerate(data["forecast"]):
        lead = index // 24 + 1
        row = report[
            (report.turbine_id == turbine)
            & (report.lead_days == lead)
            & (report.timestamp == point["timestamp"])
        ]
        assert point["normalized_power"] == pytest.approx(row.prediction.iloc[0], abs=1e-10)


@pytest.mark.parametrize("turbine", [1, 2])
def test_month_has_exact_hours_and_repeat_is_idempotent(monkeypatch, february_weather, turbine):
    monkeypatch.setattr("app.agent.fetch_archived_weather", lambda *_a, **_k: february_weather)
    for _ in range(2):
        response = client.post(f"/api/forecasts/february?turbine_id={turbine}")
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["generated_runs"] == 28
        assert len(data["forecast"]) == 672
        assert len({point["timestamp"] for point in data["forecast"]}) == 672
    rows = list(
        csv.DictReader(
            io.StringIO(client.get(f"/api/forecasts/february/export.csv?turbine_id={turbine}").text)
        )
    )
    assert len(rows) == 672
    assert rows[0]["timestamp"] == "2026-02-01T00:00:00+05:00"
    assert rows[-1]["timestamp"] == "2026-02-28T23:00:00+05:00"
    assert len(client.get("/api/forecasts?limit=100").json()) == 28


@pytest.mark.parametrize("damage", ["duplicate", "missing", "nan", "negative", "inf"])
def test_bad_weather_rejected_before_inference(february_weather, damage):
    frame = february_weather.iloc[:24].copy()
    if damage == "duplicate":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    elif damage == "missing":
        frame = frame.iloc[:-1]
    else:
        frame.loc[2, "wind_speed_100m_previous_day1"] = {
            "nan": np.nan,
            "negative": -1,
            "inf": np.inf,
        }[damage]
    with pytest.raises(ValueError):
        validate_weather(ForecastRequest(turbine_id=1, issue_date="2026-01-31"), frame)


def test_weather_timeout_returns_service_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "weather_attempts", 2)
    attempts = []

    def timeout(*_args, **_kwargs):
        attempts.append(1)
        raise httpx.ReadTimeout("test timeout")

    monkeypatch.setattr("app.services.weather.httpx.get", timeout)
    monkeypatch.setattr("app.services.weather.time.sleep", lambda _seconds: None)
    response = client.post("/api/forecasts", json={**request_body(), "data_mode": "live"})
    assert response.status_code == 503
    assert len(attempts) == 2
    assert "detail" in response.json()
    assert client.get("/api/forecasts").json() == []


def test_partial_month_is_not_exported(monkeypatch, february_weather):
    monkeypatch.setattr(
        "app.agent.fetch_archived_weather", lambda *_a, **_k: february_weather.iloc[:24]
    )
    response = client.post(
        "/api/forecasts", json={**request_body(horizon=24), "issue_date": "2026-01-31"}
    )
    assert response.status_code == 200
    assert client.get("/api/forecasts/february/export.csv?turbine_id=1").status_code == 409


def test_async_job_completes_real_offline_prediction():
    response = client.post("/api/jobs", json=request_body())
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.02)
    assert job["status"] == "completed", job
    assert len(job["result"]["forecast"]) == 48
    assert all(step["status"] == "completed" for step in job["steps"])


def test_async_job_reports_real_failed_stage():
    response = client.post("/api/jobs", json={**request_body(), "issue_date": "2027-01-01"})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == "failed":
            break
        time.sleep(0.02)
    assert job["status"] == "failed"
    assert job["steps"][0]["id"] == "weather"
    assert job["steps"][0]["status"] == "failed"
    assert all(step["status"] == "pending" for step in job["steps"][1:])


def test_features_do_not_change_with_other_target_days(february_weather):
    month = build_context_features(february_weather, 2).iloc[:24].reset_index(drop=True)
    day = build_context_features(february_weather.iloc[:24], 2).reset_index(drop=True)
    pd.testing.assert_frame_equal(month, day)
    modified = february_weather.copy()
    modified["wind_speed_100m_previous_day1"] *= 100
    pd.testing.assert_frame_equal(
        build_context_features(february_weather, 2), build_context_features(modified, 2)
    )


def test_report_metrics_recompute_from_all_predictions():
    report = json.loads((settings.model_dir / "metrics.json").read_text())
    frame = pd.read_csv(settings.model_dir / "validation_predictions.csv")
    for turbine in (1, 2):
        for lead in (1, 2):
            rows = frame[(frame.turbine_id == turbine) & (frame.lead_days == lead)]
            metric = report[f"turbine_{turbine}"][f"day_{lead}"]
            assert len(rows) == (744 if lead == 1 else 720)
            assert metric["mae"] == pytest.approx(mean_absolute_error(rows.actual, rows.prediction))
            assert metric["r2"] == pytest.approx(r2_score(rows.actual, rows.prediction))
            lower = np.clip(rows.prediction - metric["interval"]["validation_radius"], 0, 1)
            upper = np.clip(rows.prediction + metric["interval"]["validation_radius"], 0, 1)
            assert metric["interval"]["coverage"] == pytest.approx(
                ((rows.actual >= lower) & (rows.actual <= upper)).mean()
            )
            baseline_mae = mean_absolute_error(rows.actual, rows.baseline)
            assert metric["baselines"]["wind_curve"]["mae"] == pytest.approx(baseline_mae)
            assert metric["improvement_vs_baseline"]["wind_curve"][
                "mae_reduction_percent"
            ] == pytest.approx(100 * (1 - metric["mae"] / baseline_mae))


def test_validation_training_ends_before_first_forecast_issue():
    for turbine in (1, 2):
        for lead in (1, 2):
            bundle = load_model(
                settings.model_dir / "validation" / f"turbine_{turbine}_day_{lead}.joblib"
            )
            first_valid = "2026-01-01" if lead == 1 else "2026-01-02"
            first_issue = pd.Timestamp(first_valid) - pd.Timedelta(days=lead)
            assert pd.Timestamp(bundle["trained_through"]) <= first_issue
            assert pd.Timestamp(bundle["available_from"]) <= first_issue


def test_invalid_request_is_rejected():
    assert (
        client.post("/api/forecasts", json={**request_body(), "turbine_id": 3}).status_code == 422
    )
    assert (
        client.post("/api/forecasts", json={**request_body(), "horizon_hours": 72}).status_code
        == 422
    )
