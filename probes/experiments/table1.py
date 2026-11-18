"""The three classifiers and three tasks reported in Table 1."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ..data import arrays
from ..models import linear, svc
from ..models.mlp import MLPConfig, fit_predict
from ..sampling import balance, holdout

SETTINGS = {"seed", "outer_folds", "inner_folds", "c_grid_log10", "logreg_max_iter", "svc_max_iter",
            "prose_draw_seeds", "mlp_evaluation"}
TABLE = "table1"
MODELS = ("logreg", "mlp", "svc")


def tasks(data, prose_draw_seeds):
    yield "full", data, "condition", "poetry"
    yield "poetry", data[data.condition == "poetry"], "safety_label", "unsafe"
    prose = data[data.condition == "prose"]
    for seed in prose_draw_seeds:
        yield "prose", balance(prose, "safety_label", seed), "safety_label", "unsafe"


def nested_cv_scores(X, y, groups, model, settings):
    """Accuracy and AUC on each outer fold, as printed: the outer training fold is scaled
    first, then C is chosen on its inner folds. With groups, both versions of a prompt
    stay in the same fold."""
    seed, folds, inner_folds = settings["seed"], settings["outer_folds"], settings["inner_folds"]
    if groups is None:
        outer = StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y)
    else:
        outer = StratifiedGroupKFold(folds).split(X, y, groups)
    grid = np.logspace(*settings["c_grid_log10"])
    for train, test in outer:
        scaler = StandardScaler().fit(X[train])
        x, xt = scaler.transform(X[train]), scaler.transform(X[test])
        inner = (inner_folds if groups is None else
                 list(StratifiedGroupKFold(inner_folds).split(x, y[train], groups[train])))
        if model == "svc":
            fit = svc.tuned(grid, inner, settings["svc_max_iter"], seed).fit(x, y[train])
            score, predicted = fit.decision_function(xt), fit.predict(xt)
        else:
            fit = linear.tuned(grid, inner, settings["logreg_max_iter"], seed).fit(x, y[train])
            score = fit.predict_proba(xt)[:, 1]
            predicted = (score >= 0.5).astype(int)
        yield (predicted == y[test]).mean(), roc_auc_score(y[test], score)


def mlp_scores(X, y, groups, task, evaluation, config=None):
    """Accuracy on seeded holdouts. `evaluation` is a run's `mlp_evaluation` block;
    `config` replaces its model, as the A.8 grid does."""
    config = config or MLPConfig.from_settings(evaluation["model"])
    seeds = evaluation["prose_split_seeds"] if task == "prose" else evaluation["split_seeds"]
    for seed in seeds:
        dev, test = holdout(np.arange(len(y)), y, groups, evaluation["test_fraction"], seed)
        train, val = holdout(dev, y, groups, config.validation_fraction, seed)
        pred, _ = fit_predict(X, y, train, val, test, seed=seed, config=config)
        yield (y[test] == pred).mean()


def compute(settings, data):
    """One row per evaluation: outer fold for LogReg/SVC, seeded holdout for the MLP."""
    rows = []
    for task, frame, target, positive in tasks(data, settings["prose_draw_seeds"]):
        X, y = arrays(frame, target, positive)
        groups = frame.sample_id.to_numpy() if target == "condition" else None
        for model in MODELS:
            if model == "mlp":
                scores = ((acc, np.nan) for acc in mlp_scores(X, y, groups, task, settings["mlp_evaluation"]))
            else:
                scores = nested_cv_scores(X, y, groups, model, settings)
            rows.extend((task, model, accuracy, auc) for accuracy, auc in scores)
    return pd.DataFrame(rows, columns=["task", "model", "accuracy", "auc"])


def table_rows(scores):
    summary = scores.groupby(["task", "model"])[["accuracy", "auc"]].agg(["mean", "std"])
    return [[f"{value:.3f}" for value in summary.loc[task, model].dropna()]
            for model in MODELS for task in ("full", "prose", "poetry")]
