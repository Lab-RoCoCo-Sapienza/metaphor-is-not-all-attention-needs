"""The A.8 accuracy-spread check over hidden sizes and dropout."""
from collections import defaultdict
from dataclasses import replace
from itertools import product

import numpy as np

from ..data import arrays
from ..models.mlp import MLPConfig
from .format_and_safety import mlp_scores, tasks

SETTINGS = {"hidden_grid", "dropout_grid", "prose_draw_seeds", "mlp_evaluation"}


def compute(settings, data):
    """Per task, the range of mean accuracy across the grid."""
    evaluation = settings["mlp_evaluation"]
    base = MLPConfig.from_settings(evaluation["model"])
    means = defaultdict(list)
    for hidden, dropout in product(settings["hidden_grid"], settings["dropout_grid"]):
        config = replace(base, hidden=tuple(hidden), dropout=dropout)
        scores = defaultdict(list)
        for task, frame, target, positive in tasks(data, settings["prose_draw_seeds"]):
            X, y = arrays(frame, target, positive)
            groups = frame.sample_id.to_numpy() if target == "condition" else None
            scores[task].extend(mlp_scores(X, y, groups, task, evaluation, config))
        for task, runs in scores.items():
            means[task].append(np.mean(runs))
    return {task: float(np.ptp(values)) for task, values in means.items()}


def summary(spreads):
    return ("A.8 accuracy spread across MLP configurations (paper bound ≤ 0.04): "
            + ", ".join(f"{task} {value:.5f}" for task, value in spreads.items()))
