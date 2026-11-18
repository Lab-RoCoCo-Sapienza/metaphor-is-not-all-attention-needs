"""A.11 persona contrasts using the original folds and checkpoint rule."""
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data import arrays
from ..models import linear, svc
from ..models.mlp import MLPConfig, fit_predict
from ..sampling import balance

SETTINGS = {"pooled_row", "seed", "folds", "C", "logreg_max_iter", "mlp"}
INPUTS = ("pooled_row",)
TABLE = "a11"
MODELS = ("logreg", "mlp", "svc")


def contrasts(frame, seed):
    for persona in ("skilled", "unskilled"):
        yield f"within_{persona}", frame[frame.persona == persona], "condition"
    for persona in ("skilled", "unskilled"):
        restricted = frame[(frame.condition == "poetry") | (frame.persona == persona)]
        yield f"vs_prose_{persona}", balance(restricted, "condition", seed), "condition"
    yield "persona_only", frame[frame.condition == "prose"], "persona"


def compute(settings, data, pooled_row):
    """Mean and spread of accuracy per (contrast, model). `pooled_row` holds Table 1's scores,
    whose format probe the manuscript repeats as the first row."""
    seed, config = settings["seed"], MLPConfig.from_settings(settings["mlp"])
    pooled = pooled_row[pooled_row.task == "full"].groupby("model").accuracy.agg(["mean", "std"])
    rows = [("pooled", model, *pooled.loc[model]) for model in MODELS]
    for name, frame, target in contrasts(data, seed):
        X, y = arrays(frame, target, "prose" if target == "condition" else "unskilled")
        scores = {model: [] for model in MODELS}
        for train, test in StratifiedKFold(settings["folds"], shuffle=True, random_state=seed).split(X, y):
            for model in MODELS:
                if model == "mlp":
                    # In the original A.11 fits the scoring fold also selected the checkpoint.
                    pred, _ = fit_predict(X, y, train, test, test, seed=seed, config=config)
                    accuracy = (pred == y[test]).mean(dtype=np.float32).item()
                else:
                    clf = (svc.fixed(settings["C"]) if model == "svc" else
                           linear.fixed(settings["C"], settings["logreg_max_iter"]))
                    pred = make_pipeline(StandardScaler(), clf).fit(X[train], y[train]).predict(X[test])
                    accuracy = (pred == y[test]).mean()
                scores[model].append(accuracy)
        rows.extend((name, model, np.mean(scores[model]), np.std(scores[model])) for model in MODELS)
    return pd.DataFrame(rows, columns=["contrast", "model", "mean", "std"])


def table_rows(scores):
    def shown(contrast, value):
        if contrast == "pooled":
            return f"{value:.3f}"
        # The manuscript transcribes the four-decimal persona summary at three decimals.
        return str(Decimal(f"{value:.4f}").quantize(Decimal(".001"), rounding=ROUND_HALF_UP))

    return [[shown(contrast, value) for value in block[["mean", "std"]].to_numpy().ravel()]
            for contrast, block in scores.groupby("contrast", sort=False)]
