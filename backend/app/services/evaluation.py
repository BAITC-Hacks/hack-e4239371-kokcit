"""Report honest baseline differences and empirical residual-band coverage."""

import numpy as np


def baseline_improvement(metrics: dict) -> dict:
    return {
        name: {
            "mae_reduction": baseline["mae"] - metrics["mae"],
            "mae_reduction_percent": 100 * (1 - metrics["mae"] / baseline["mae"])
            if baseline["mae"] > 0
            else None,
            "rmse_reduction_percent": 100 * (1 - metrics["rmse"] / baseline["rmse"])
            if baseline["rmse"] > 0
            else None,
            "r2_delta": metrics["r2"] - baseline["r2"],
        }
        for name, baseline in metrics["baselines"].items()
    }


def interval_report(actual, prediction, validation_radius: float, production_radius: float) -> dict:
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    lower = np.clip(prediction - validation_radius, 0, 1)
    upper = np.clip(prediction + validation_radius, 0, 1)
    return {
        "method": "absolute_residual_quantile",
        "label": "Empirical error range; not a guaranteed confidence interval",
        "nominal_level": 0.9,
        "validation_radius": float(validation_radius),
        "calibration_period": "2025-12-01 / 2025-12-31",
        "calibration_also_used_for_model_selection": True,
        "evaluation_period": "2026-01-01 / 2026-01-31",
        "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_width": float(np.mean(upper - lower)),
        "samples": len(actual),
        "production_radius": float(production_radius),
        "production_calibration_period": "2026-01-01 / 2026-01-31",
        "production_coverage": None,
        "note": "January coverage uses December radius. February coverage is unknown without targets.",
    }
