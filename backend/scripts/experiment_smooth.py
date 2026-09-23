"""December-only selection of smooth and physically structured power regressors."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from app.services.dataset import load_turbine_csv
from app.services.features import FEATURE_COLUMNS
from scripts.experiment_xgboost import build_frame, split
from scripts.train_models import evaluation_cutoff, fit_selected, make_frame, score


def make_estimator(config):
    if config["family"] == "svr":
        return make_pipeline(
            StandardScaler(),
            SVR(C=config["C"], gamma=config["gamma"], epsilon=0.035, cache_size=512),
        )
    return HistGradientBoostingRegressor(
        max_iter=350,
        max_leaf_nodes=config["leaves"],
        learning_rate=0.04,
        l2_regularization=5,
        random_state=42,
    )


def configs():
    for mode in ("basic", "compact"):
        for gamma in (0.02, 0.1):
            for constant in (0.3, 3.0):
                yield {"family": "svr", "mode": mode, "C": constant, "gamma": gamma, "months": 12}
    for mode in ("advanced", "trajectory"):
        for leaves in (7, 15, 31):
            for months in (12, 24):
                yield {"family": "hist", "mode": mode, "leaves": leaves, "months": months}


def columns_for(mode, all_columns):
    if mode == "basic":
        return [
            name for name in FEATURE_COLUMNS if name not in ("wind_100m_squared", "wind_100m_cubed")
        ]
    if mode == "compact":
        return [
            "wind_10m_ms",
            "wind_100m_ms",
            "temperature_c",
            "wind_direction_sin",
            "wind_direction_cos",
            "wind_100m_ms_daily_mean",
            "wind_100m_ms_daily_std",
            "wind_u_daily_mean",
            "wind_v_daily_mean",
            "wind_100m_ms_trend_6",
        ]
    return all_columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data_cache/smooth"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    deployed_selection = json.loads(Path("models/model_selection.json").read_text())
    report = {}
    for turbine in (1, 2):
        observed = load_turbine_csv(next(args.data_dir.glob(f"*turbine {turbine}.csv")), turbine)
        weather = pd.read_csv(
            f"data_cache/weather_turbine_{turbine}.csv", parse_dates=["timestamp"]
        )
        for lead in (1, 2):
            key = f"turbine_{turbine}_day_{lead}"
            frames = {
                mode: build_frame(observed, weather, lead, mode)
                for mode in ("advanced", "trajectory")
            }
            candidates, predictions = [], []
            for config in configs():
                started = time.perf_counter()
                frame, all_columns = frames[
                    "trajectory" if config["mode"] == "trajectory" else "advanced"
                ]
                columns = columns_for(config["mode"], all_columns)
                training = split(frame, evaluation_cutoff("2025-12-01", lead), config["months"])
                december = frame[frame.timestamp.between("2025-12-01", "2025-12-31 23:00")]
                model = make_estimator(config)
                model.fit(training[columns], training.power)
                prediction = np.clip(model.predict(december[columns]), 0, 1)
                candidate = {
                    "config": config,
                    "december": score(december.power, prediction),
                    "seconds": time.perf_counter() - started,
                }
                candidates.append(candidate)
                predictions.append(prediction)
                print(json.dumps({"key": key, **candidate}), flush=True)
                report[key] = {"candidates": candidates, "status": "searching_december"}
                (args.output / "report.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )

            baseline_config = deployed_selection[key]["winner"]
            baseline_frame, baseline_columns = make_frame(
                observed, weather, lead, baseline_config["context"]
            )
            baseline = fit_selected(
                baseline_config,
                baseline_frame,
                baseline_columns,
                evaluation_cutoff("2025-12-01", lead),
            )
            december = baseline_frame[
                baseline_frame.timestamp.between("2025-12-01", "2025-12-31 23:00")
            ]
            baseline_prediction = np.clip(baseline.predict(december[baseline_columns]), 0, 1)
            baseline_score = score(december.power, baseline_prediction)
            options = [{"candidate": None, "weight": 0, "december": baseline_score}]
            for index, prediction in enumerate(predictions):
                for weight in (0.25, 0.5, 0.75, 1):
                    blended = weight * prediction + (1 - weight) * baseline_prediction
                    options.append(
                        {
                            "candidate": index,
                            "weight": weight,
                            "december": score(december.power, blended),
                        }
                    )
            winner = min(
                (
                    option
                    for option in options
                    if option["december"]["mae"] <= baseline_score["mae"]
                ),
                key=lambda option: option["december"]["rmse"],
            )
            report[key].update(
                winner=winner, current_december=baseline_score, status="december_selection_frozen"
            )
            (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if winner["candidate"] is None:
                continue
            config = candidates[winner["candidate"]]["config"]
            frame, all_columns = frames[
                "trajectory" if config["mode"] == "trajectory" else "advanced"
            ]
            columns = columns_for(config["mode"], all_columns)
            training = split(frame, evaluation_cutoff("2026-01-01", lead), config["months"])
            start = "2026-01-01" if lead == 1 else "2026-01-02"
            january = frame[frame.timestamp.between(start, "2026-01-31 23:00")]
            model = make_estimator(config)
            model.fit(training[columns], training.power)
            prediction = np.clip(model.predict(january[columns]), 0, 1)
            old = pd.read_csv("models/validation_predictions.csv")
            old = old[(old.turbine_id == turbine) & (old.lead_days == lead)]
            if not np.allclose(january.power, old.actual):
                raise ValueError("January evaluation rows differ")
            blended = (
                winner["weight"] * prediction + (1 - winner["weight"]) * old.prediction.to_numpy()
            )
            report[key].update(
                january=score(january.power, blended),
                current_january=score(old.actual, old.prediction),
                status="evaluated",
            )
            (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(
                json.dumps({"key": key, "winner": winner, "january": report[key]["january"]}),
                flush=True,
            )


if __name__ == "__main__":
    main()
