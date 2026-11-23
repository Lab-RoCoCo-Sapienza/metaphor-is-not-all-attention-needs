"""Fit the A.12 safety coefficients and render the two safe-up paper charts."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..charts import render_chart
from ..data import FEATURES, arrays
from ..models import linear
from ..sampling import balance

SETTINGS = {"seed", "folds", "C", "max_iter", "y_step"}


def compute(settings, data):
    """Per format, the mean and spread across folds of each raw-feature coefficient."""
    seed, coefficients = settings["seed"], {}
    for format_name in ("poetry", "prose"):
        frame = data[data.condition == format_name]
        if format_name == "prose":
            frame = balance(frame, "safety_label", seed)
        X, y = arrays(frame, "safety_label", "unsafe", dtype=np.float64)
        coefs = []
        for train, _ in StratifiedKFold(settings["folds"], shuffle=True, random_state=seed).split(X, y):
            model = linear.fixed(settings["C"], settings["max_iter"], class_weight="balanced")
            coefs.append(model.fit(X[train], y[train]).coef_[0])
        weights = np.stack(coefs)
        coefficients[format_name] = pd.DataFrame({"feature_name": FEATURES, "beta_mean": weights.mean(axis=0),
                                                  "beta_std": weights.std(axis=0)})
    return coefficients


def render(coefficients, settings, folder):
    for format_name, table in coefficients.items():
        render_chart(table, folder / f"logistic_features_{format_name}_predict-safe.png",
                     up_class="safe", down_class="unsafe", positive_class="unsafe", red_class="unsafe",
                     y_step=settings["y_step"][format_name])
