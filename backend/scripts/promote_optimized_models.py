"""Build a reproducible CPU-serving release from frozen December search finalists.

January is used as a development-validation regression gate, not a blind test.
Native CUDA predictions must match portable inference before any artifact is published.
"""

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.dataset import load_turbine_csv
from app.services.evaluation import baseline_improvement, interval_report
from app.services.features import build_model_features
from app.services.model import ForecastEnsemble, load_model, save_model
from app.services.portable_boost import PortableBoostRegressor
from scripts.experiment_smooth import columns_for, make_estimator
from scripts.experiment_xgboost import build_frame, fit, split
from scripts.train_models import evaluation_cutoff, fit_selected, make_frame, score

VERSION = "windflow-v3"


def dump(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--extra-site-packages", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path("data_cache/pre_v3_models"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    if args.extra_site_packages:
        sys.path.append(str(args.extra_site_packages.resolve()))
    import xgboost as xgb

    # Snapshot the release once. Re-running promotion never recursively blends v3.
    if not args.base_dir.exists():
        shutil.copytree(args.model_dir, args.base_dir)
    base_metrics = json.loads((args.base_dir / "metrics.json").read_text())
    base_selection = json.loads((args.base_dir / "model_selection.json").read_text())
    predictions = pd.read_csv(args.base_dir / "validation_predictions.csv")
    metrics = copy.deepcopy(base_metrics)
    searches = {
        family: json.loads(Path(f"data_cache/{family}/report.json").read_text())
        for family in ("xgboost", "smooth")
    }
    staging = Path("data_cache/v3_release")
    (staging / "validation").mkdir(parents=True, exist_ok=True)
    audit = {
        "version": VERSION,
        "method": "December chooses each family's configuration and blend weight; January is a development-validation regression gate. Retain the previous model if neither MAE nor RMSE improves together.",
        "january_is_blind_test": False,
        "gpu_library": xgb.__version__,
        "gpu_configurations": sum(len(item["candidates"]) for item in searches["xgboost"].values()),
        "smooth_configurations": sum(
            len(item["candidates"]) for item in searches["smooth"].values()
        ),
        "models": {},
    }
    for turbine in (1, 2):
        observed = load_turbine_csv(next(args.data_dir.glob(f"*turbine {turbine}.csv")), turbine)
        weather = pd.read_csv(
            f"data_cache/weather_turbine_{turbine}.csv", parse_dates=["timestamp"]
        )
        weather = weather[weather.timestamp <= "2026-01-31 23:00"].reset_index(drop=True)
        for lead in (1, 2):
            key = f"turbine_{turbine}_day_{lead}"
            filename = f"{key}.joblib"
            old_metric = base_metrics[f"turbine_{turbine}"][f"day_{lead}"]
            eligible = []
            for family, records in searches.items():
                candidate = records[key]
                evaluated = candidate.get("january")
                if (
                    evaluated
                    and evaluated["mae"] < old_metric["mae"]
                    and evaluated["rmse"] < old_metric["rmse"]
                ):
                    eligible.append((family, candidate))
            old_validation = load_model(args.base_dir / "validation" / filename)
            old_production = load_model(args.base_dir / filename)
            if not eligible:
                for directory, bundle in (
                    (staging / "validation", old_validation),
                    (staging, old_production),
                ):
                    save_model({**bundle, "model_version": VERSION}, directory / filename)
                audit["models"][key] = {
                    "promoted": False,
                    "reason": "No finalist improved both January MAE and RMSE",
                    "metrics": old_metric,
                }
                continue
            family, experiment = min(
                eligible, key=lambda entry: entry[1]["winner"]["december"]["rmse"]
            )
            winner = experiment["winner"]
            candidate = experiment["candidates"][winner["candidate"]]
            config = candidate["config"]
            mode = (
                config["mode"]
                if config["mode"] in ("context", "advanced", "trajectory")
                else "advanced"
            )
            frame, all_columns = build_frame(observed, weather, lead, mode)
            columns = (
                all_columns if family == "xgboost" else columns_for(config["mode"], all_columns)
            )
            december = frame[frame.timestamp.between("2025-12-01", "2025-12-31 23:00")]
            january = frame[
                frame.timestamp.between(old_metric["validation_start"], "2026-01-31 23:00")
            ]
            cutoff = evaluation_cutoff("2026-01-01", lead)
            training = split(frame, cutoff, config["months"])
            production_training = split(frame, "2026-02-01", config["months"])
            native_difference = None
            if family == "xgboost":
                validation_path = Path(f"data_cache/xgboost/{key}_validation.json")
                december_path = Path(f"data_cache/xgboost/{key}_{winner['candidate']}.json")
                validation_model = PortableBoostRegressor.from_file(validation_path)
                december_model = PortableBoostRegressor.from_file(december_path)
                native = xgb.Booster()
                native.load_model(validation_path)
                native_prediction = native.predict(xgb.DMatrix(january[columns].astype(np.float32)))
                native_difference = float(
                    np.max(np.abs(native_prediction - validation_model.predict(january[columns])))
                )
                if native_difference > 1e-6:
                    raise ValueError("Portable inference differs from XGBoost")
                # Refit identical frozen hyperparameters through January for February serving.
                production_booster = fit(
                    xgb,
                    config,
                    frame,
                    columns,
                    lead,
                    validation=True,
                    iterations=candidate["iterations"],
                    training_cutoff="2026-02-01",
                )
                production_path = staging / f"{key}_native.json"
                production_booster.save_model(production_path)
                production_model = PortableBoostRegressor.from_file(production_path)
            else:
                validation_model = make_estimator(config).fit(training[columns], training.power)
                december_training = split(
                    frame, evaluation_cutoff("2025-12-01", lead), config["months"]
                )
                december_model = make_estimator(config).fit(
                    december_training[columns], december_training.power
                )
                production_model = make_estimator(config).fit(
                    production_training[columns], production_training.power
                )

            base_frame, base_columns = make_frame(
                observed, weather, lead, old_validation["feature_mode"] == "context"
            )
            base_december = base_frame[
                base_frame.timestamp.between("2025-12-01", "2025-12-31 23:00")
            ]
            base_december_model = fit_selected(
                base_selection[key]["winner"],
                base_frame,
                base_columns,
                evaluation_cutoff("2025-12-01", lead),
            )
            weight = winner["weight"]
            december_prediction = (1 - weight) * np.clip(
                base_december_model.predict(base_december[base_columns]), 0, 1
            ) + weight * np.clip(december_model.predict(december[columns]), 0, 1)
            radius = float(np.quantile(np.abs(december.power - december_prediction), 0.9))
            old_columns = list(
                build_model_features(weather, lead, old_validation["feature_mode"]).columns
            )
            validation_ensemble = ForecastEnsemble(
                [(old_validation["estimator"], old_columns), (validation_model, columns)],
                [1 - weight, weight],
                clip_members=True,
            )
            production_ensemble = ForecastEnsemble(
                [(old_production["estimator"], old_columns), (production_model, columns)],
                [1 - weight, weight],
                clip_members=True,
            )
            prediction = validation_ensemble.predict(january[all_columns])
            metric = score(january.power, prediction)
            if not (metric["mae"] < old_metric["mae"] and metric["rmse"] < old_metric["rmse"]):
                raise ValueError("Promotion regression gate failed during reproducibility check")
            mask = (predictions.turbine_id == turbine) & (predictions.lead_days == lead)
            if not np.allclose(predictions.loc[mask, "actual"], january.power):
                raise ValueError("Validation targets or order changed")
            predictions.loc[mask, "prediction"] = prediction
            production_radius = float(np.quantile(np.abs(january.power - prediction), 0.9))
            for directory, previous, estimator, error_radius in (
                (staging / "validation", old_validation, validation_ensemble, radius),
                (staging, old_production, production_ensemble, production_radius),
            ):
                save_model(
                    {
                        **previous,
                        "estimator": estimator,
                        "feature_mode": mode,
                        "model_version": VERSION,
                        "error_radius": error_radius,
                    },
                    directory / filename,
                )
            updated = {
                **old_metric,
                **metric,
                "model_profile": f"v3_{family}_blend",
                "feature_mode": mode,
                "error_radius": production_radius,
            }
            updated["previous_version"] = {
                name: old_metric[name] for name in ("mae", "rmse", "r2", "samples", "model_profile")
            }
            updated["previous_version"]["same_evaluation_rows"] = True
            updated["improvement_vs_baseline"] = baseline_improvement(updated)
            updated["interval"] = interval_report(
                january.power, prediction, radius, production_radius
            )
            updated["interval"]["evaluation_period"] = (
                f"{old_metric['validation_start']} / 2026-01-31"
            )
            updated["interval"]["production_calibration_period"] = (
                f"{old_metric['validation_start']} / 2026-01-31"
            )
            updated["target_met"] = metric["r2"] >= 0.8
            updated["training_components"] = {
                "previous_model": old_metric["train_samples"],
                family: len(training),
            }
            updated["selection_period"] = (
                "December 2025 tuning; January 2026 development-validation promotion gate"
            )
            metrics[f"turbine_{turbine}"][f"day_{lead}"] = updated
            audit["models"][key] = {
                "promoted": True,
                "family": family,
                "config": config,
                "blend_weight": weight,
                "december": score(december.power, december_prediction),
                "before": {name: old_metric[name] for name in ("mae", "rmse", "r2")},
                "after": metric,
                "native_portable_max_difference": native_difference,
                "base_artifact_sha256": hashlib.sha256(
                    (args.base_dir / filename).read_bytes()
                ).hexdigest(),
            }
            print(json.dumps({"key": key, **audit["models"][key]}), flush=True)

    metrics["_meta"].update(
        model_version=VERSION,
        selection=audit["method"],
        january_is_blind_test=False,
        evaluation_caveat="January is a development-validation period used for promotion checks; independent February quality requires February observations.",
        optimization_report="optimization_report.json",
    )
    dump(staging / "metrics.json", metrics)
    dump(staging / "optimization_report.json", audit)
    dump(staging / "metrics_before_v3.json", base_metrics)
    predictions.to_csv(staging / "validation_predictions.csv", index=False, lineterminator="\n")
    # All four model sets and reports exist before any published artifact changes.
    paths = [Path(f"turbine_{t}_day_{d}.joblib") for t in (1, 2) for d in (1, 2)]
    paths += [Path("validation") / path for path in paths.copy()]
    paths += [
        Path(name)
        for name in (
            "metrics.json",
            "optimization_report.json",
            "metrics_before_v3.json",
            "validation_predictions.csv",
        )
    ]
    for relative in paths:
        destination = args.model_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(staging / relative, temporary)
        temporary.replace(destination)
    print("Published windflow-v3: prediction runs do not require XGBoost or CUDA", flush=True)


if __name__ == "__main__":
    main()
