import hashlib
import json

import pytest

from app.catalog import TURBINES
from app.config import settings
from app.services.weather import WeatherUnavailable, fetch_archived_weather
from scripts.prepare_demo import prepare_archive


@pytest.mark.parametrize("manifest_name", ["manifest.json", "february_manifest.json"])
def test_bundled_snapshot_bytes_match_manifest_on_every_platform(manifest_name):
    manifest = json.loads((settings.demo_dir / manifest_name).read_text(encoding="utf-8"))
    for record in manifest["files"]:
        content = (settings.demo_dir / record["path"]).read_bytes()
        assert b"\r\n" not in content, "Bundled weather must retain LF line endings"
        assert hashlib.sha256(content).hexdigest() == record["sha256"]


@pytest.mark.parametrize("turbine", [1, 2])
def test_bundled_real_february_archive(turbine):
    frame = fetch_archived_weather(*TURBINES[turbine], "2026-02-01", "2026-02-28", mode="offline")
    assert frame.attrs["source"] == "bundled_archive"
    assert frame.attrs["manifest_verified"]
    assert frame.attrs["provider_url"] == "https://previous-runs-api.open-meteo.com/v1/forecast"
    assert len(frame) == frame.timestamp.nunique() == 672
    assert str(frame.timestamp.min()) == "2026-02-01 00:00:00"
    assert str(frame.timestamp.max()) == "2026-02-28 23:00:00"
    assert not frame.isna().any().any()


def test_prepared_february_archive_has_provenance_and_both_turbines(
    tmp_path, monkeypatch, february_weather
):
    monkeypatch.setattr(settings, "demo_dir", tmp_path / "demo")
    weather = february_weather.copy()
    weather.attrs["source"] = "open_meteo"  # Mock network boundary; never publish this fixture.
    monkeypatch.setattr("scripts.prepare_demo.fetch_archived_weather", lambda *_a, **_kw: weather)
    manifest = prepare_archive()
    assert manifest["publication_time_verified"] is False
    assert manifest["archive_kind"] == "fixed_lead_offsets"
    assert manifest["first_issue_date"] == "2026-01-31"
    assert manifest["last_issue_date"] == "2026-02-27"
    assert len(manifest["files"]) == 2
    for record in manifest["files"]:
        assert record["hours"] == 672
        path = settings.demo_dir / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        turbine = record["turbine_id"]
        frame = fetch_archived_weather(
            *TURBINES[turbine], "2026-02-01", "2026-02-28", mode="offline"
        )
        assert len(frame) == 672
        assert frame.timestamp.is_unique
        assert frame.attrs["source"] == "bundled_archive"
        assert frame.attrs["manifest_verified"]
        assert frame.attrs["snapshot_sha256"] == record["sha256"]
    saved = json.loads((settings.demo_dir / "february_manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest

    # A finite, structurally valid change must still fail the provenance check.
    path = settings.demo_dir / manifest["files"][0]["path"]
    path.write_bytes(path.read_bytes().replace(b"20.0", b"21.0", 1))
    with pytest.raises(WeatherUnavailable):
        fetch_archived_weather(*TURBINES[1], "2026-02-01", "2026-02-28", mode="offline")


def test_second_download_failure_does_not_replace_existing_archive(
    tmp_path, monkeypatch, february_weather
):
    monkeypatch.setattr(settings, "demo_dir", tmp_path)
    original = tmp_path / "weather_february_turbine_1.csv"
    original.write_text("existing", encoding="utf-8")
    calls = []

    def fetch(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise WeatherUnavailable("offline")
        frame = february_weather.copy()
        frame.attrs["source"] = "open_meteo"
        return frame

    monkeypatch.setattr("scripts.prepare_demo.fetch_archived_weather", fetch)
    with pytest.raises(WeatherUnavailable):
        prepare_archive()
    assert original.read_text() == "existing"
    assert not (tmp_path / "february_manifest.json").exists()


def test_archive_preparation_rejects_non_network_source(tmp_path, monkeypatch, february_weather):
    monkeypatch.setattr(settings, "demo_dir", tmp_path)
    monkeypatch.setattr(
        "scripts.prepare_demo.fetch_archived_weather", lambda *_a, **_kw: february_weather
    )
    with pytest.raises(ValueError, match="настоящий ответ"):
        prepare_archive()
