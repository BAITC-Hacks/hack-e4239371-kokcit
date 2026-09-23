"""Download real archived forecasts for a reproducible offline February demo."""

import argparse
import hashlib
import json
from datetime import UTC, datetime

import pandas as pd

from app.catalog import TURBINES
from app.config import settings
from app.services.weather import fetch_archived_weather


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-02-28")
    args = parser.parse_args()
    settings.demo_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": settings.open_meteo_base_url,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "files": [],
    }
    for turbine, coordinates in TURBINES.items():
        weather = fetch_archived_weather(
            *coordinates, args.start, args.end, mode="live", on_event=print
        )
        expected = pd.date_range(args.start, f"{args.end} 23:00", freq="h")
        if not pd.DatetimeIndex(weather.timestamp).equals(expected) or weather.isna().any().any():
            raise ValueError("Неполный архив погоды, демонстрационный файл не сохранён")
        path = settings.demo_dir / f"weather_turbine_{turbine}.csv"
        weather.to_csv(path, index=False)
        manifest["files"].append(
            {
                "path": path.name,
                "hours": len(weather),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        print(f"Saved {path.name}: {len(weather)} hours", flush=True)
    (settings.demo_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
