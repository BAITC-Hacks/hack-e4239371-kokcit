"""Recompute report additions from saved January predictions without retraining."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.dataset import load_turbine_csv
from app.services.evaluation import baseline_improvement, interval_report
from app.services.model import load_model, save_model
from scripts.train_models import make_frame, score


def refresh(model_dir: Path, raw_paths: dict, cache_dir: Path):
    path = model_dir / "metrics.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.read_csv(model_dir / "validation_predictions.csv")
    audited_rows = []
    for turbine in (1, 2):
        observed = weather = None
        if "baseline_mean" not in rows:
            if not raw_paths.get(turbine):
                raise ValueError(
                    "Provide --turbine-1 and --turbine-2 to reconstruct the training-mean baseline once"
                )
            observed = load_turbine_csv(raw_paths[turbine], turbine)
            weather = pd.read_csv(
                cache_dir / f"weather_turbine_{turbine}.csv", parse_dates=["timestamp"]
            )
        for lead in (1, 2):
            data = rows[(rows.turbine_id == turbine) & (rows.lead_days == lead)].copy()
            metric = report[f"turbine_{turbine}"][f"day_{lead}"]
            recomputed = score(data.actual, data.prediction)
            if any(abs(metric[key] - recomputed[key]) > 1e-9 for key in ("mae", "rmse", "r2")):
                raise ValueError("Saved metrics do not match saved predictions")
            validation = load_model(model_dir / metric["validation_model_path"])
            production = load_model(model_dir / metric["model_path"])
            if "baseline_mean" not in data:
                frame, _columns = make_frame(
                    observed, weather, lead, metric["feature_mode"] == "context"
                )
                training = frame[
                    frame.timestamp.between(
                        metric["training_start"], f"{validation['trained_through']} 23:00"
                    )
                ]
                if len(training) != metric["train_samples"]:
                    raise ValueError(
                        "Reconstructed training window does not match the saved report"
                    )
                data["baseline_mean"] = float(training.power.mean())
            validation_start = "2026-01-01" if lead == 1 else "2026-01-02"
            data = data[data.timestamp >= validation_start].copy()
            expected_samples = 744 if lead == 1 else 720
            if len(data) != expected_samples or data.timestamp.nunique() != expected_samples:
                raise ValueError(f"Expected {expected_samples} unique validation hours")
            metric.update(score(data.actual, data.prediction))
            metric["baselines"] = {
                "wind_curve": score(data.actual, data.baseline),
                "mean": score(data.actual, data.baseline_mean),
            }
            metric["validation_start"] = validation_start
            metric["available_from"] = "2025-12-31"
            metric["target_met"] = metric["r2"] >= metric["target_r2"]
            if "previous_version" in metric:
                metric["previous_version"]["same_evaluation_rows"] = lead == 1
            production_radius = float(np.quantile(np.abs(data.actual - data.prediction), 0.9))
            metric["error_radius"] = production_radius
            save_model(
                {**validation, "available_from": "2025-12-31"},
                model_dir / metric["validation_model_path"],
            )
            save_model(
                {**production, "available_from": "2026-01-31", "error_radius": production_radius},
                model_dir / metric["model_path"],
            )
            metric["improvement_vs_baseline"] = baseline_improvement(metric)
            metric["interval"] = interval_report(
                data.actual, data.prediction, validation["error_radius"], production_radius
            )
            metric["interval"]["evaluation_period"] = f"{validation_start} / 2026-01-31"
            metric["interval"]["production_calibration_period"] = f"{validation_start} / 2026-01-31"
            audited_rows.append(data)
            print(
                json.dumps(
                    {
                        "turbine": turbine,
                        "lead": lead,
                        **{key: metric[key] for key in ("mae", "rmse", "r2", "samples")},
                        "coverage": metric["interval"]["coverage"],
                        "mae_improvement_percent": metric["improvement_vs_baseline"]["wind_curve"][
                            "mae_reduction_percent"
                        ],
                    }
                ),
                flush=True,
            )
    report["_meta"].update(
        {
            "evaluation": "January: day 1 Jan 1-31 (744 hours), day 2 Jan 2-31 (720 hours). January 1 day 2 excluded because selection/calibration completed Dec 31, later than its Dec 30 issue.",
            "horizon_definition": {
                "day_1": "hours 1-24",
                "day_2": "hours 25-48, not the full 48-hour aggregate",
            },
            "publication_time_verified": False,
            "january_seen_in_earlier_experiments": True,
            "evaluation_caveat": "Current selection uses December only, but January was inspected in earlier development. It is not a newly sealed blind test.",
        }
    )
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    pd.concat(audited_rows).to_csv(
        model_dir / "validation_predictions.csv", index=False, lineterminator="\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--turbine-1", type=Path)
    parser.add_argument("--turbine-2", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    args = parser.parse_args()
    refresh(args.model_dir, {1: args.turbine_1, 2: args.turbine_2}, args.cache_dir)


if __name__ == "__main__":
    main()
