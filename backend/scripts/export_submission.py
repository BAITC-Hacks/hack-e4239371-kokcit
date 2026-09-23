"""Generate complete February CSVs for both turbines and an audit manifest."""

import argparse
import hashlib
import json
from pathlib import Path

from app.agent import run_february
from app.config import settings
from app.services.reporting import Report, export_pdf, export_xlsx
from app.services.storage import february_forecast_to_csv, get_february_forecasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--reports", action="store_true", help="Also generate PDF and XLSX reports")
    parser.add_argument("--output", type=Path, default=Path("../artifacts/submission"))
    args = parser.parse_args()
    results = {}
    manifest = {"period": "2026-02-01 / 2026-02-28", "files": []}
    for turbine in (1, 2):
        result = run_february(
            turbine,
            "offline" if args.offline else "auto",
            on_progress=lambda done, total: print(f"{done}/{total}", flush=True),
        )
        content = february_forecast_to_csv(turbine)
        if content is None:
            raise ValueError("No complete forecast was generated")
        results[turbine] = content
        manifest["files"].append(
            {
                "name": f"turbine_{turbine}_february.csv",
                "rows": len(result.forecast),
                "model_version": result.model_version,
                "data_source": result.data_source,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        )
    reports = {}
    if args.reports:
        for turbine in (1, 2):
            report = Report(get_february_forecasts(turbine), monthly=True)
            for extension, exporter in (("pdf", export_pdf), ("xlsx", export_xlsx)):
                name = f"turbine_{turbine}_february.{extension}"
                content = exporter(report)
                reports[name] = content
                manifest["files"].append(
                    {
                        "name": name,
                        "rows": len(report.points),
                        "model_version": report.runs[0].model_version,
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
    # Publish files only after both full months have been generated successfully.
    args.output.mkdir(parents=True, exist_ok=True)
    for turbine, content in results.items():
        (args.output / f"turbine_{turbine}_february.csv").write_text(
            content, encoding="utf-8", newline=""
        )
    for name, content in reports.items():
        (args.output / name).write_bytes(content)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output / "metrics.json").write_bytes((settings.model_dir / "metrics.json").read_bytes())
    print(f"Submission ready: {args.output.resolve()}")


if __name__ == "__main__":
    main()
