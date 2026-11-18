"""Paper split and sampling rules. Row order is part of the protocol."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def balance(frame, column, seed):
    """Undersample every larger class to the size of the smallest, keeping row order."""
    counts = frame[column].value_counts()
    size = int(counts.min())
    rng = np.random.default_rng(seed)
    selected = []
    for label in counts.index:
        indices = np.flatnonzero(frame[column].to_numpy() == label)
        if len(indices) > size:
            indices = rng.choice(indices, size=size, replace=False)
        selected.extend(indices)
    return frame.iloc[sorted(selected)]


def holdout(indices, y, groups, fraction, seed):
    """Split off a stratified fraction; with groups, keep each group on one side."""
    if groups is None:
        return train_test_split(indices, test_size=fraction, stratify=y[indices], random_state=seed)
    split = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    train, heldout = next(split.split(indices, groups=groups[indices]))
    return indices[train], indices[heldout]
