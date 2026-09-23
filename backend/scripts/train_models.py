"""Chronological selection (December), evaluation (January), then production refit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.catalog import LOCAL_TIMEZONE, MODEL_VERSION
from app.services.dataset import load_turbine_csv
from app.services.evaluation import baseline_improvement, interval_report
from app.services.features import FEATURE_COLUMNS, build_context_features, build_features
from app.services.model import ForecastEnsemble, save_model

PROFILES = {
    "small": {"max_iter": 350, "max_leaf_nodes": 15, "learning_rate": 0.04, "l2_regularization": 2},
    "large": {"max_iter": 450, "max_leaf_nodes": 31, "learning_rate": 0.05, "l2_regularization": 1},
    "extra": {"n_estimators": 120, "min_samples_leaf": 8, "max_features": 0.8, "n_jobs": 1},
    "extra_deep": {"n_estimators": 120, "min_samples_leaf": 3, "max_features": 1.0, "n_jobs": 1},
}


def score(actual, prediction):
    prediction = np.clip(prediction, 0, 1)
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2": float(r2_score(actual, prediction)),
        "samples": len(actual),
    }


def make_model(profile):
    factory = ExtraTreesRegressor if profile.startswith("extra") else HistGradientBoostingRegressor
    return factory(random_state=42, **PROFILES[profile])


def make_frame(observed, weather, lead, context):
    features = (build_context_features if context else build_features)(weather, lead)
    columns = list(features.columns)
    features["timestamp"] = weather.timestamp
    frame = features.merge(observed[["timestamp", "power"]], on="timestamp", how="inner")
    return frame.dropna(subset=columns + ["power"]), columns


def training_window(frame, end, months):
    return frame[
        (frame.timestamp < end)
        & (frame.timestamp >= pd.Timestamp(end) - pd.DateOffset(months=months))
    ]


def evaluation_cutoff(first_valid_day, lead):
    # First valid hour minus lead_days gives the issue date; training may include
    # that date but nothing after it, including at the start of an evaluation month.
    return str((pd.Timestamp(first_valid_day) - pd.Timedelta(days=lead - 1)).date())


def fit_selected(chosen, frame, columns, end):
    if chosen["profile"] == "ensemble":
        members = []
        for config in chosen["members"]:
            member_columns = columns if config["context"] else list(FEATURE_COLUMNS)
            data = training_window(frame, end, config["months"])
            model = make_model(config["profile"])
            model.fit(data[member_columns], data.power)
            members.append((model, member_columns))
        return ForecastEnsemble(members, chosen["weights"])
    model = make_model(chosen["profile"])
    data = training_window(frame, end, chosen["months"])
    model.fit(data[columns], data.power)
    return model


def select_ensemble(observed, weather, lead, selection):
    """Compare diverse blends exclusively on December; no January labels read here."""
    frame, columns = make_frame(observed, weather, lead, True)
    tune = frame[frame.timestamp.between("2025-12-01", "2025-12-31 23:00")]
    singles = [
        candidate for candidate in selection["candidates"] if candidate["profile"] != "ensemble"
    ]
    best_hist = min(
        (c for c in singles if not c["profile"].startswith("extra")),
        key=lambda c: c["december"]["rmse"],
    )
    best_extra = min(
        (c for c in singles if c["profile"].startswith("extra")),
        key=lambda c: c["december"]["rmse"],
    )
    predictions = []
    for config in (best_hist, best_extra):
        member_columns = columns if config["context"] else list(FEATURE_COLUMNS)
        model = fit_selected(config, frame, member_columns, evaluation_cutoff("2025-12-01", lead))
        predictions.append(model.predict(tune[member_columns]))
    for weight in (0.25, 0.5, 0.75):
        metrics = score(tune.power, weight * predictions[0] + (1 - weight) * predictions[1])
        selection["candidates"].append(
            {
                "profile": "ensemble",
                "context": True,
                "months": max(best_hist["months"], best_extra["months"]),
                "members": [
                    {key: config[key] for key in ("profile", "context", "months")}
                    for config in (best_hist, best_extra)
                ],
                "weights": [weight, 1 - weight],
                "december": metrics,
            }
        )
    selection["winner"] = min(selection["candidates"], key=lambda c: c["december"]["rmse"])
    return selection


def select(observed, weather, lead):
    candidates = []
    for context in (False, True):
        frame, columns = make_frame(observed, weather, lead, context)
        tune = frame[frame.timestamp.between("2025-12-01", "2025-12-31 23:00")]
        for months in (24, 12, 6):
            train = training_window(frame, evaluation_cutoff("2025-12-01", lead), months)
            for profile in PROFILES:
                model = make_model(profile)
                model.fit(train[columns], train.power)
                metrics = score(tune.power, model.predict(tune[columns]))
                candidates.append(
                    {"context": context, "months": months, "profile": profile, "december": metrics}
                )
    return select_ensemble(
        observed,
        weather,
        lead,
        {
            "winner": min(candidates, key=lambda c: c["december"]["rmse"]),
            "candidates": candidates,
        },
    )


def train_one(observed, weather, turbine, lead, selection, model_dir):
    chosen = selection["winner"]
    frame, columns = make_frame(observed, weather, lead, chosen["context"])
    validation_cutoff = evaluation_cutoff("2026-01-01", lead)
    training = training_window(frame, validation_cutoff, chosen["months"])
    # Hyperparameters and the error band use all of December. Their earliest
    # issue date is December 31, so day-2 January 1 must not enter evaluation.
    validation_start = "2026-01-01" if lead == 1 else "2026-01-02"
    january = frame[frame.timestamp.between(validation_start, "2026-01-31 23:00")]
    expected_samples = 744 if lead == 1 else 720
    if len(january) != expected_samples:
        raise ValueError(f"Expected {expected_samples} validation hours, found {len(january)}")
    model = fit_selected(chosen, frame, columns, validation_cutoff)
    prediction = np.clip(model.predict(january[columns]), 0, 1)
    metrics = score(january.power, prediction)
    mean_prediction = np.full(len(january), training.power.mean())
    curve = IsotonicRegression(out_of_bounds="clip").fit(training.wind_100m_ms, training.power)
    curve_prediction = curve.predict(january.wind_100m_ms)
    baselines = {
        "mean": score(january.power, mean_prediction),
        "wind_curve": score(january.power, curve_prediction),
    }

    # December residuals describe uncertainty for the January validation model.
    december = frame[frame.timestamp.between("2025-12-01", "2025-12-31 23:00")]
    calibration_model = fit_selected(chosen, frame, columns, evaluation_cutoff("2025-12-01", lead))
    december_errors = np.abs(
        december.power - np.clip(calibration_model.predict(december[columns]), 0, 1)
    )
    radius = float(np.quantile(december_errors, 0.9))
    bundle = {
        "estimator": model,
        "feature_mode": "context" if chosen["context"] else "basic",
        "trained_through": str((pd.Timestamp(validation_cutoff) - pd.Timedelta(days=1)).date()),
        "error_radius": radius,
        "model_version": MODEL_VERSION,
        "available_from": "2025-12-31",
    }
    name = f"turbine_{turbine}_day_{lead}.joblib"
    save_model(bundle, model_dir / "validation" / name)
    production = training_window(frame, "2026-02-01", chosen["months"])
    final_model = fit_selected(chosen, frame, columns, "2026-02-01")
    january_radius = float(np.quantile(np.abs(january.power - prediction), 0.9))
    save_model(
        {
            **bundle,
            "estimator": final_model,
            "trained_through": "2026-01-31",
            "available_from": "2026-01-31",
            "error_radius": january_radius,
        },
        model_dir / name,
    )

    report = {
        **metrics,
        "model_profile": chosen["profile"],
        "feature_mode": bundle["feature_mode"],
        "train_samples": len(training),
        "production_train_samples": len(production),
        "training_start": str(training.timestamp.min()),
        "trained_through": bundle["trained_through"],
        "validation_start": validation_start,
        "validation_end": "2026-01-31",
        "selection_period": "2025-12-01 / 2025-12-31",
        "baselines": baselines,
        "error_radius": january_radius,
        "model_path": name,
        "validation_model_path": f"validation/{name}",
        "target_r2": 0.8,
        "target_met": metrics["r2"] >= 0.8,
    }
    report["improvement_vs_baseline"] = baseline_improvement(report)
    report["interval"] = interval_report(january.power, prediction, radius, january_radius)
    report["interval"]["evaluation_period"] = f"{validation_start} / 2026-01-31"
    report["interval"]["production_calibration_period"] = f"{validation_start} / 2026-01-31"
    report["available_from"] = "2025-12-31"
    points = pd.DataFrame(
        {
            "timestamp": [
                value.to_pydatetime().replace(tzinfo=LOCAL_TIMEZONE).isoformat()
                for value in january.timestamp
            ],
            "turbine_id": turbine,
            "lead_days": lead,
            "actual": january.power.to_numpy(),
            "prediction": prediction,
            "baseline": curve_prediction,
            "baseline_mean": mean_prediction,
        }
    )
    return report, points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turbine-1", type=Path, required=True)
    parser.add_argument("--turbine-2", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--try-ensembles",
        action="store_true",
        help="Compare blends when reusing a selection report",
    )
    parser.add_argument(
        "--selection-report",
        type=Path,
        help="Reuse a previously generated December-only comparison",
    )
    args = parser.parse_args()
    selections = json.loads(args.selection_report.read_text()) if args.selection_report else {}
    report = {
        "_meta": {
            "model_version": MODEL_VERSION,
            "selection": "December 2025 RMSE; January targets are not used for fitting or selection",
            "evaluation": "January 2026: day 1 uses Jan 1-31 (744 hours); day 2 uses Jan 2-31 (720 hours), because selection completed Dec 31",
            "production_refit": "Through 2026-01-31, used only for forecasts issued from January 31",
            "features": "Archived weather forecasts and calendar only; no future measured wind/power",
            "weather": "Open-Meteo Previous Runs, fixed lead offsets (not a single initialization)",
            "interval": "Descriptive 90th percentile of absolute validation residuals, not guaranteed coverage",
            "horizon_definition": {
                "day_1": "hours 1-24",
                "day_2": "hours 25-48, not the full 48-hour aggregate",
            },
            "publication_time_verified": False,
            "raw_data": {},
        }
    }
    frames = []
    args.model_dir.mkdir(parents=True, exist_ok=True)
    old_metrics = args.model_dir / "previous_metrics.json"
    current_metrics = args.model_dir / "metrics.json"
    if current_metrics.exists() and not old_metrics.exists():
        old_metrics.write_bytes(current_metrics.read_bytes())
    for turbine, path in ((1, args.turbine_1), (2, args.turbine_2)):
        observed = load_turbine_csv(path, turbine)
        weather_path = args.cache_dir / f"weather_turbine_{turbine}.csv"
        if not weather_path.exists():
            from scripts.train_baseline import load_or_fetch_weather

            load_or_fetch_weather(turbine, args.cache_dir)
        weather = pd.read_csv(weather_path, parse_dates=["timestamp"])
        weather = weather[weather.timestamp <= "2026-01-31 23:00"].reset_index(drop=True)
        report["_meta"]["raw_data"][f"turbine_{turbine}"] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "weather_sha256": hashlib.sha256(weather_path.read_bytes()).hexdigest(),
        }
        report[f"turbine_{turbine}"] = {}
        for lead in (1, 2):
            key = f"turbine_{turbine}_day_{lead}"
            if key not in selections:
                selections[key] = select(observed, weather, lead)
            elif args.try_ensembles:
                selections[key] = select_ensemble(observed, weather, lead, selections[key])
            metrics, points = train_one(
                observed, weather, turbine, lead, selections[key], args.model_dir
            )
            report[f"turbine_{turbine}"][f"day_{lead}"] = metrics
            frames.append(points)
            print(json.dumps({"turbine": turbine, "lead": lead, **metrics}), flush=True)
    if old_metrics.exists():
        previous = json.loads(old_metrics.read_text(encoding="utf-8"))
        for turbine in (1, 2):
            for lead in (1, 2):
                report[f"turbine_{turbine}"][f"day_{lead}"]["previous_version"] = previous[
                    f"turbine_{turbine}"
                ][f"day_{lead}"]
    current_metrics.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    # Only December scores are needed to reproduce selection; exclude exploratory January scores.
    for selection in selections.values():
        selection["winner"].pop("january", None)
    (args.model_dir / "model_selection.json").write_text(
        json.dumps(selections, indent=2), encoding="utf-8"
    )
    pd.concat(frames).to_csv(args.model_dir / "validation_predictions.csv", index=False)


if __name__ == "__main__":
    main()
