import argparse
import json
from pathlib import Path

import pandas as pd

from app.services.dataset import load_turbine_csv
from app.services.features import FEATURE_COLUMNS, build_features
from app.services.model import MODEL_PROFILES, create_baseline_model, evaluate, save_model
from app.services.weather import fetch_archived_weather

TURBINES = {
    1: (43.645150, 78.535604),
    2: (43.643198, 78.538828),
}


def load_or_fetch_weather(turbine_id: int, cache_dir: Path) -> pd.DataFrame:
    cache_path = cache_dir / f"weather_turbine_{turbine_id}.csv"
    if cache_path.exists():
        weather = pd.read_csv(cache_path)
        weather["timestamp"] = pd.to_datetime(weather["timestamp"])
        return weather

    latitude, longitude = TURBINES[turbine_id]
    weather = fetch_archived_weather(latitude, longitude, "2024-01-01", "2026-01-31")
    cache_dir.mkdir(parents=True, exist_ok=True)
    weather.to_csv(cache_path, index=False)
    return weather


def training_frame(turbine: pd.DataFrame, weather: pd.DataFrame, lead_days: int) -> pd.DataFrame:
    features = build_features(weather, lead_days)
    features["timestamp"] = weather["timestamp"].to_numpy()
    joined = features.merge(turbine[["timestamp", "power"]], on="timestamp", how="inner")
    return joined.dropna(subset=[*FEATURE_COLUMNS, "power"])


def train_one(turbine_id: int, csv_path: Path, cache_dir: Path, model_dir: Path) -> dict:
    turbine = load_turbine_csv(csv_path, turbine_id)
    weather = load_or_fetch_weather(turbine_id, cache_dir)
    results = {}

    for lead_days in (1, 2):
        frame = training_frame(turbine, weather, lead_days)
        train = frame[frame["timestamp"] < "2026-01-01"]
        validation = frame[frame["timestamp"] >= "2026-01-01"]

        candidates = []
        for profile in MODEL_PROFILES:
            candidate = create_baseline_model(profile)
            candidate.fit(train[list(FEATURE_COLUMNS)], train["power"])
            candidate_metrics = evaluate(candidate, validation)
            candidates.append((candidate_metrics["mae"], profile, candidate_metrics))
        _, best_profile, metrics = min(candidates, key=lambda item: item[0])

        final_model = create_baseline_model(best_profile)
        final_model.fit(frame[list(FEATURE_COLUMNS)], frame["power"])
        model_path = model_dir / f"turbine_{turbine_id}_day_{lead_days}.joblib"
        save_model(final_model, model_path)
        results[f"day_{lead_days}"] = {
            **metrics,
            "model_profile": best_profile,
            "train_samples": len(train),
            "model_path": str(model_path),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WindFlow baseline models")
    parser.add_argument("--turbine-1", type=Path, required=True)
    parser.add_argument("--turbine-2", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    report = {
        "turbine_1": train_one(1, args.turbine_1, args.cache_dir, args.model_dir),
        "turbine_2": train_one(2, args.turbine_2, args.cache_dir, args.model_dir),
    }
    args.model_dir.mkdir(parents=True, exist_ok=True)
    (args.model_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
