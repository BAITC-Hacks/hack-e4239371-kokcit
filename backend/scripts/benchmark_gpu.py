"""Optional CatBoost GPU comparison. Does not replace the verified application models."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.dataset import load_turbine_csv
from scripts.train_models import evaluation_cutoff, make_frame, score, training_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        from catboost import CatBoostRegressor
        from catboost.utils import get_gpu_device_count
    except ImportError:
        parser.exit(2, "Install optional CatBoost first: uv pip install catboost\n")
    if get_gpu_device_count() < 1:
        parser.exit(2, "CatBoost cannot access a CUDA GPU. Check the NVIDIA driver.\n")
    output = Path("data_cache/gpu")
    output.mkdir(parents=True, exist_ok=True)
    current = json.loads(Path("models/metrics.json").read_text())
    report = {
        "device": "GPU:0",
        "selection": "December RMSE",
        "evaluation": "January 2026",
        "note": "GPU floating point reductions can be nondeterministic. Application models are not replaced.",
        "models": {},
    }
    for turbine in (1, 2):
        matches = list(args.data_dir.glob(f"*turbine {turbine}.csv"))
        if len(matches) != 1:
            raise ValueError(f"Expected one turbine {turbine}.csv file in {args.data_dir}")
        observed = load_turbine_csv(matches[0], turbine)
        weather = pd.read_csv(
            f"data_cache/weather_turbine_{turbine}.csv", parse_dates=["timestamp"]
        )
        for lead in (1, 2):
            frame, columns = make_frame(observed, weather, lead, True)
            tune = frame[frame.timestamp.between("2025-12-01", "2025-12-31 23:00")]
            candidates = []
            for depth in (4, 6, 8):
                train = training_window(frame, evaluation_cutoff("2025-12-01", lead), 24)
                parameters = {
                    "depth": depth,
                    "iterations": 1200,
                    "learning_rate": 0.05,
                    "loss_function": "RMSE",
                    "l2_leaf_reg": 5,
                    "task_type": "GPU",
                    "devices": "0",
                    "random_seed": 42,
                    "allow_writing_files": False,
                }
                model = CatBoostRegressor(**parameters)
                model.fit(
                    train[columns],
                    train.power,
                    eval_set=(tune[columns], tune.power),
                    early_stopping_rounds=100,
                    verbose=False,
                )
                metric = score(tune.power, model.predict(tune[columns]))
                candidates.append(
                    {
                        "parameters": {
                            **parameters,
                            "iterations": max(1, model.get_best_iteration() + 1),
                        },
                        "december": metric,
                    }
                )
                print(
                    json.dumps(
                        {"turbine": turbine, "lead": lead, "depth": depth, "december": metric}
                    ),
                    flush=True,
                )
            best = min(candidates, key=lambda candidate: candidate["december"]["rmse"])
            cutoff = evaluation_cutoff("2026-01-01", lead)
            train = training_window(frame, cutoff, 24)
            validation_start = "2026-01-01" if lead == 1 else "2026-01-02"
            january = frame[frame.timestamp.between(validation_start, "2026-01-31 23:00")]
            model = CatBoostRegressor(**best["parameters"])
            model.fit(train[columns], train.power, verbose=False)
            prediction = np.clip(model.predict(january[columns]), 0, 1)
            metrics = score(january.power, prediction)
            key = f"turbine_{turbine}_day_{lead}"
            model.save_model(str(output / f"{key}.cbm"))
            report["models"][key] = {
                "gpu_january": metrics,
                "current_cpu_january": current[f"turbine_{turbine}"][f"day_{lead}"],
                "selection": best,
                "training_end_exclusive": cutoff,
                "validation_start": validation_start,
                "available_from": "2025-12-31",
            }
            (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(
                json.dumps({"turbine": turbine, "lead": lead, "gpu_january": metrics}), flush=True
            )
    print(f"GPU comparison saved: {output / 'comparison.json'}")


if __name__ == "__main__":
    main()
