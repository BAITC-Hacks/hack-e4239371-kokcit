"""NumPy inference for scalar, numerical XGBoost gbtree squared-error exports.

Training uses XGBoost CUDA. Serving needs only NumPy and the exported arrays.
Other objectives, categorical trees and vector targets are rejected explicitly.
"""

import json
from pathlib import Path

import numpy as np


class PortableBoostRegressor:
    def __init__(self, document: dict):
        learner = document["learner"]
        booster = learner["gradient_booster"]
        params = learner["learner_model_param"]
        if learner["objective"]["name"] != "reg:squarederror" or booster["name"] != "gbtree":
            raise ValueError("Portable inference supports squared-error gbtree models only")
        if int(params["num_target"]) != 1 or int(params["num_class"]) != 0:
            raise ValueError("Portable inference requires one regression target")
        self.feature_names_in_ = np.asarray(learner["feature_names"])
        self.n_features_in_ = int(params["num_feature"])
        self.base_score = np.float32(np.atleast_1d(json.loads(params["base_score"]))[0])
        self.trees = []
        for tree in booster["model"]["trees"]:
            if any(tree.get("split_type", [])):
                raise ValueError("Categorical splits are not supported")
            self.trees.append(
                (
                    np.asarray(tree["left_children"], dtype=np.int32),
                    np.asarray(tree["right_children"], dtype=np.int32),
                    np.asarray(tree["split_indices"], dtype=np.int32),
                    np.asarray(tree["split_conditions"], dtype=np.float32),
                    np.asarray(tree["default_left"], dtype=bool),
                )
            )

    @classmethod
    def from_file(cls, path: Path):
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def predict(self, features):
        values = (
            features[list(self.feature_names_in_)].to_numpy(dtype=np.float32)
            if hasattr(features, "columns")
            else np.asarray(features, dtype=np.float32)
        )
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Incorrect feature shape for portable model")
        prediction = np.full(len(values), self.base_score, dtype=np.float32)
        for left, right, columns, conditions, default_left in self.trees:
            nodes = np.zeros(len(values), dtype=np.int32)
            for _ in range(len(left)):
                active = np.flatnonzero(left[nodes] >= 0)
                if not len(active):
                    break
                current = nodes[active]
                value = values[active, columns[current]]
                go_left = np.where(
                    np.isnan(value), default_left[current], value < conditions[current]
                )
                nodes[active] = np.where(go_left, left[current], right[current])
            else:
                raise ValueError("Malformed tree: traversal did not reach a leaf")
            prediction += conditions[nodes]
        return prediction.astype(float)
