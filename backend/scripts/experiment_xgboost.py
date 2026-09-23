"""GPU model search: December selection, January evaluation, no app-model replacement."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.advanced_features import build_advanced_features
from app.services.dataset import load_turbine_csv
from app.services.features import build_context_features
from scripts.train_models import evaluation_cutoff, fit_selected, make_frame, score


def configurations():
    rows = []
    for mode in ("context", "advanced", "trajectory"):
        for months in (12, 24):
            for depth in (3, 5, 7):
                rows.append(
                    {
                        "mode": mode,
                        "months": months,
                        "depth": depth,
                        "loss": "reg:squarederror",
                        "half_life": None,
                    }
                )
    for months in (12, 24):
        for depth in (3, 5):
            for loss in ("reg:absoluteerror", "reg:pseudohubererror"):
                rows.append(
                    {
                        "mode": "advanced",
                        "months": months,
                        "depth": depth,
                        "loss": loss,
                        "half_life": None,
                    }
                )
    for depth in (3, 5):
        rows.append(
            {
                "mode": "trajectory",
                "months": 24,
                "depth": depth,
                "loss": "reg:squarederror",
                "half_life": 180,
            }
        )
    return rows


def build_frame(observed, weather, lead, mode):
    features = (
        build_context_features(weather, lead)
        if mode == "context"
        else build_advanced_features(weather, lead, mode == "trajectory")
    )
    columns = list(features.columns)
    frame = features.assign(timestamp=weather.timestamp).merge(
        observed[["timestamp", "power"]], on="timestamp", how="inner"
    )
    return frame.dropna(subset=columns + ["power"]), columns


def split(frame, end, months):
    return frame[
        (frame.timestamp < end)
        & (frame.timestamp >= pd.Timestamp(end) - pd.DateOffset(months=months))
    ]


def fit(xgb, config, frame, columns, lead, validation=False, iterations=None, training_cutoff=None):
    cutoff = training_cutoff or evaluation_cutoff(
        "2026-01-01" if validation else "2025-12-01", lead
    )
    training = split(frame, cutoff, config["months"])
    tune = frame[frame.timestamp.between("2025-12-01", "2025-12-31 23:00")]
    weights = None
    if config["half_life"]:
        age_days = (pd.Timestamp(cutoff) - training.timestamp).dt.total_seconds() / 86400
        weights = np.exp2(-age_days / config["half_life"])
    dtrain = xgb.DMatrix(training[columns].astype(np.float32), label=training.power, weight=weights)
    params = {
        "device": "cuda:0",
        "tree_method": "hist",
        "objective": config["loss"],
        "max_depth": config["depth"],
        "min_child_weight": 30,
        "eta": 0.035,
        "reg_lambda": 10,
        "reg_alpha": 0.1,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "eval_metric": "rmse",
        "seed": 42,
        "nthread": 4,
    }
    if config["loss"] == "reg:pseudohubererror":
        params.update(huber_slope=0.15, base_score=0.4)
    kwargs = (
        {}
        if validation
        else {
            "evals": [
                (xgb.DMatrix(tune[columns].astype(np.float32), label=tune.power), "december")
            ],
            "early_stopping_rounds": 80,
        }
    )
    booster = xgb.train(
        params, dtrain, num_boost_round=iterations or 1800, verbose_eval=False, **kwargs
    )
    if not validation:
        booster = booster[: booster.best_iteration + 1]
    configuration = json.loads(booster.save_config())
    if not configuration["learner"]["generic_param"]["device"].startswith("cuda"):
        raise RuntimeError("XGBoost did not use CUDA; refusing to claim a GPU experiment")
    return booster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--extra-site-packages", type=Path)
    parser.add_argument("--turbines", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--leads", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", type=Path, default=Path("data_cache/xgboost"))
    args = parser.parse_args()
    if args.extra_site_packages:
        sys.path.append(str(args.extra_site_packages.resolve()))
    import xgboost as xgb

    args.output.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps({"xgboost": xgb.__version__, "cuda_build": xgb.build_info().get("USE_CUDA")}),
        flush=True,
    )
    selection = json.loads(Path("models/model_selection.json").read_text(encoding="utf-8"))
    report_path = args.output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for turbine in args.turbines:
        paths = list(args.data_dir.glob(f"*turbine {turbine}.csv"))
        if len(paths) != 1:
            raise ValueError("Expected one source CSV per turbine")
        observed = load_turbine_csv(paths[0], turbine)
        weather = pd.read_csv(
            f"data_cache/weather_turbine_{turbine}.csv", parse_dates=["timestamp"]
        )
        weather = weather[weather.timestamp <= "2026-01-31 23:00"].reset_index(drop=True)
        for lead in args.leads:
            key = f"turbine_{turbine}_day_{lead}"
            frames = {
                mode: build_frame(observed, weather, lead, mode)
                for mode in ("context", "advanced", "trajectory")
            }
            candidates = []
            for index, config in enumerate(configurations()):
                started = time.perf_counter()
                frame, columns = frames[config["mode"]]
                tune = frame[frame.timestamp.between("2025-12-01", "2025-12-31 23:00")]
                booster = fit(xgb, config, frame, columns, lead)
                prediction = np.clip(
                    booster.predict(xgb.DMatrix(tune[columns].astype(np.float32))), 0, 1
                )
                metrics = score(tune.power, prediction)
                candidate = {
                    "config": config,
                    "december": metrics,
                    "iterations": booster.num_boosted_rounds(),
                    "seconds": time.perf_counter() - started,
                }
                booster.save_model(str(args.output / f"{key}_{index}.json"))
                np.save(args.output / f"{key}_{index}_december.npy", prediction)
                candidates.append(candidate)
                print(json.dumps({"key": key, "candidate": index, **candidate}), flush=True)
                report[key] = {"candidates": candidates, "status": "searching_december"}
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            # Reconstruct the deployed baseline's December prediction for honest blending.
            current_config = selection[key]["winner"]
            current_frame, current_columns = make_frame(
                observed, weather, lead, current_config["context"]
            )
            current_december = current_frame[
                current_frame.timestamp.between("2025-12-01", "2025-12-31 23:00")
            ]
            current_model = fit_selected(
                current_config,
                current_frame,
                current_columns,
                evaluation_cutoff("2025-12-01", lead),
            )
            current_prediction = np.clip(
                current_model.predict(current_december[current_columns]), 0, 1
            )
            current_score = score(current_december.power, current_prediction)
            options = [{"candidate": None, "weight": 0.0, "december": current_score}]
            for index, candidate in enumerate(candidates):
                prediction = np.load(args.output / f"{key}_{index}_december.npy")
                for weight in (0.25, 0.5, 0.75, 1.0):
                    metric = score(
                        current_december.power,
                        weight * prediction + (1 - weight) * current_prediction,
                    )
                    options.append({"candidate": index, "weight": weight, "december": metric})
            eligible = [
                option for option in options if option["december"]["mae"] <= current_score["mae"]
            ]
            winner = min(eligible, key=lambda option: option["december"]["rmse"])
            report[key] = {
                "candidates": candidates,
                "winner": winner,
                "current_december": current_score,
                "status": "december_selection_frozen",
            }
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({"key": key, "winner": winner}), flush=True)
            if winner["candidate"] is None:
                continue
            candidate = candidates[winner["candidate"]]
            frame, columns = frames[candidate["config"]["mode"]]
            start = "2026-01-01" if lead == 1 else "2026-01-02"
            january = frame[frame.timestamp.between(start, "2026-01-31 23:00")]
            booster = fit(
                xgb,
                candidate["config"],
                frame,
                columns,
                lead,
                validation=True,
                iterations=candidate["iterations"],
            )
            prediction = np.clip(
                booster.predict(xgb.DMatrix(january[columns].astype(np.float32))), 0, 1
            )
            old_rows = pd.read_csv("models/validation_predictions.csv")
            old_rows = old_rows[(old_rows.turbine_id == turbine) & (old_rows.lead_days == lead)]
            if not np.allclose(january.power, old_rows.actual):
                raise ValueError("January comparison rows differ")
            blended = (
                winner["weight"] * prediction
                + (1 - winner["weight"]) * old_rows.prediction.to_numpy()
            )
            report[key].update(
                january=score(january.power, blended),
                current_january=score(old_rows.actual, old_rows.prediction),
                status="evaluated",
            )
            booster.save_model(str(args.output / f"{key}_validation.json"))
            pd.DataFrame(
                {
                    "timestamp": old_rows.timestamp,
                    "actual": january.power.to_numpy(),
                    "prediction": blended,
                    "gpu_prediction": prediction,
                    "current": old_rows.prediction,
                }
            ).to_csv(args.output / f"{key}_january.csv", index=False)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "key": key,
                        "january": report[key]["january"],
                        "current": report[key]["current_january"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
