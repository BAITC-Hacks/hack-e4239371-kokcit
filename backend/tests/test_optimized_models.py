import numpy as np
import pandas as pd
import pytest

from app.services.features import build_model_features
from app.services.model import ForecastEnsemble
from app.services.portable_boost import PortableBoostRegressor


def exported_tree():
    return {
        "learner": {
            "feature_names": ["wind"],
            "learner_model_param": {
                "base_score": "[0.5]",
                "num_target": "1",
                "num_class": "0",
                "num_feature": "1",
            },
            "objective": {"name": "reg:squarederror"},
            "gradient_booster": {
                "name": "gbtree",
                "model": {
                    "trees": [
                        {
                            "left_children": [1, -1, -1],
                            "right_children": [2, -1, -1],
                            "split_indices": [0, 0, 0],
                            "split_conditions": [1.0, -0.2, 0.3],
                            "default_left": [1, 0, 0],
                            "split_type": [0, 0, 0],
                        }
                    ]
                },
            },
        }
    }


def test_portable_tree_preserves_threshold_equality_and_missing_branch():
    model = PortableBoostRegressor(exported_tree())
    rows = pd.DataFrame({"wind": [0.999, 1.0, 2.0, np.nan], "ignored": [99, 99, 99, 99]})
    np.testing.assert_allclose(model.predict(rows), [0.3, 0.8, 0.8, 0.3], atol=1e-7)


def test_portable_model_rejects_unsupported_objective():
    document = exported_tree()
    document["learner"]["objective"]["name"] = "binary:logistic"
    with pytest.raises(ValueError, match="squared-error"):
        PortableBoostRegressor(document)


@pytest.mark.parametrize("mode", ["advanced", "trajectory"])
@pytest.mark.parametrize("lead", [1, 2])
def test_new_features_only_use_the_current_target_day(february_weather, mode, lead):
    whole = build_model_features(february_weather, lead, mode).iloc[:24].reset_index(drop=True)
    daily = build_model_features(february_weather.iloc[:24], lead, mode).reset_index(drop=True)
    pd.testing.assert_frame_equal(whole, daily)
    assert np.isfinite(daily.to_numpy()).all()
    if lead == 2:
        altered = february_weather.copy()
        for column in altered:
            if column.endswith("previous_day1"):
                altered[column] = 9999
        pd.testing.assert_frame_equal(
            build_model_features(february_weather, lead, mode),
            build_model_features(altered, lead, mode),
        )


def test_blending_clips_each_member_before_averaging():
    class Constant:
        def __init__(self, value):
            self.value = value

        def predict(self, values):
            return np.full(len(values), self.value)

    blend = ForecastEnsemble(
        [(Constant(-1), ["wind"]), (Constant(0.5), ["wind"])], [0.5, 0.5], clip_members=True
    )
    assert blend.predict(pd.DataFrame({"wind": [1]}))[0] == pytest.approx(0.25)
