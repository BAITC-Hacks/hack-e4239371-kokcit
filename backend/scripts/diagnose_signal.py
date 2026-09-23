"""December-only diagnostics. The measured-wind oracle is explicitly not deployable."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from app.services.dataset import load_turbine_csv
from scripts.train_models import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for turbine in (1, 2):
        observed = load_turbine_csv(next(args.data_dir.glob(f"*turbine {turbine}.csv")), turbine)
        weather = pd.read_csv(
            f"data_cache/weather_turbine_{turbine}.csv", parse_dates=["timestamp"]
        )
        merged = observed.merge(weather, on="timestamp", how="inner")
        train = merged[merged.timestamp < "2025-12-01"]
        december = merged[merged.timestamp.between("2025-12-01", "2025-12-31 23:00")]
        oracle = IsotonicRegression(out_of_bounds="clip").fit(train.measured_wind_ms, train.power)
        rows = {
            "not_deployable_measured_wind_oracle": score(
                december.power, oracle.predict(december.measured_wind_ms)
            ),
            "warning": "Oracle uses future measured wind only to diagnose the problem, never for production or claimed forecast metrics.",
            "training_hours": len(train),
            "december_hours": len(december),
            "out_of_range_targets": int((~observed.power.between(0, 1)).sum()),
            "low_power_despite_high_measured_wind_fraction": float(
                ((december.measured_wind_ms > 8) & (december.power < 0.1)).mean()
            ),
            "leads": {},
        }
        for lead in (1, 2):
            forecast_wind = december[f"wind_speed_100m_previous_day{lead}"] / 3.6
            shifts = []
            indexed_weather = (
                weather.set_index("timestamp")[f"wind_speed_100m_previous_day{lead}"] / 3.6
            )
            for offset in range(-12, 13):
                values = indexed_weather.reindex(
                    december.timestamp + pd.Timedelta(hours=offset)
                ).to_numpy()
                shifts.append(
                    {
                        "offset": offset,
                        "correlation": float(np.corrcoef(values, december.measured_wind_ms)[0, 1]),
                    }
                )
            rows["leads"][str(lead)] = {
                "forecast_wind_vs_measured_wind_correlation": float(
                    forecast_wind.corr(december.measured_wind_ms)
                ),
                "forecast_wind_vs_power_correlation": float(forecast_wind.corr(december.power)),
                "shift_diagnostic_only": shifts,
                "shift_note": "Offsets are diagnostic only; no shifted weather or altered timestamps are deployed.",
            }
        report[f"turbine_{turbine}"] = rows
        print(json.dumps({"turbine": turbine, **rows}), flush=True)
    Path("data_cache/signal_diagnostics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
