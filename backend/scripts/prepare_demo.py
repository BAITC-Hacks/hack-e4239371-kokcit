"""Download real archived forecasts for a reproducible offline February demo."""

import argparse
import hashlib
import json
from datetime import UTC, datetime

import pandas as pd

from app.catalog import TURBINES
from app.config import settings
from app.services.weather import (
    TIMING_NOTE,
    _requested_variables,
    fetch_archived_weather,
    validate_archive,
)


def prepare_archive(start="2026-02-01", end="2026-02-28"):
    """Validate both real downloads before replacing any bundled data."""
    is_february = (start, end) == ("2026-02-01", "2026-02-28")
    manifest = {
        "source": settings.open_meteo_base_url,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "start_date": start,
        "end_date": end,
        "timezone": "Asia/Almaty",
        "wind_speed_unit": "kmh",
        "variables": _requested_variables().split(","),
        "archive_kind": "fixed_lead_offsets",
        "publication_time_verified": False,
        "timing_note": TIMING_NOTE,
        "files": [],
    }
    if is_february:
        manifest.update(first_issue_date="2026-01-31", last_issue_date="2026-02-27")
    pending = []
    for turbine, coordinates in TURBINES.items():
        weather = fetch_archived_weather(*coordinates, start, end, mode="live", on_event=print)
        if weather.attrs.get("source") != "open_meteo":
            raise ValueError("Для подготовки архива требуется настоящий ответ Open-Meteo")
        validate_archive(weather, start, end)
        columns = ["timestamp", *_requested_variables().split(",")]
        content = weather[columns].to_csv(index=False, lineterminator="\n").encode("utf-8")
        prefix = "weather_february" if is_february else "weather"
        path = settings.demo_dir / f"{prefix}_turbine_{turbine}.csv"
        pending.append((path, content))
        manifest["files"].append(
            {
                "path": path.name,
                "turbine_id": turbine,
                "latitude": coordinates[0],
                "longitude": coordinates[1],
                "hours": len(weather),
                "sha256": hashlib.sha256(content).hexdigest(),
                "downloaded_at": datetime.now(UTC).isoformat(),
            }
        )
    settings.demo_dir.mkdir(parents=True, exist_ok=True)
    manifest_name = "february_manifest.json" if is_february else "manifest.json"
    pending.append(
        (
            settings.demo_dir / manifest_name,
            json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
        )
    )
    for path, content in pending:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        print(f"Saved {path.name}: {len(content)} bytes", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-02-01")
    parser.add_argument("--end", default="2026-02-28")
    args = parser.parse_args()
    pd.Timestamp(args.start)  # Validate dates before network calls.
    pd.Timestamp(args.end)
    prepare_archive(args.start, args.end)


if __name__ == "__main__":
    main()
