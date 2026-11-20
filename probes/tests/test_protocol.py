"""The cleaned pool and the split rules every run depends on."""
from dataclasses import replace

import numpy as np

from probes.data import FEATURES, arrays, load_data
from probes.experiments.persona import contrasts
from probes.experiments.table1 import tasks
from probes.models.mlp import MLPConfig, fit_predict
from probes.runner import Runner
from probes.sampling import balance, holdout


def test_cleaned_pool():
    data = load_data()
    assert data.shape[0] == 2303 and len(FEATURES) == 72
    assert data.groupby(["condition", "safety_label"]).size().tolist() == [614, 538, 1029, 122]
    assert data.persona.isin(["skilled", "unskilled"]).all()


def test_balanced_prose_draws_and_persona_subsets():
    data = load_data()
    seeds = Runner().runs["table1"]["prose_draw_seeds"]
    sizes = [(task, len(frame)) for task, frame, _, _ in tasks(data, seeds)]
    assert sizes == [("full", 2303), ("poetry", 1152)] + [("prose", 244)] * 5
    prose = data[data.condition == "prose"]
    first, again = balance(prose, "safety_label", 42), balance(prose, "safety_label", 42)
    assert first.index.equals(again.index) and first.index.is_monotonic_increasing
    assert first.safety_label.value_counts().to_dict() == {"safe": 122, "unsafe": 122}
    assert [len(frame) for _, frame, _ in contrasts(data, 42)] == [1148, 1155, 1146, 1156, 1151]


def test_format_holdouts_keep_both_versions_of_a_prompt_together():
    data = load_data()
    _, y = arrays(data, "condition", "poetry")
    groups = data.sample_id.to_numpy()
    dev, test = holdout(np.arange(len(y)), y, groups, 0.15, 7)
    train, val = holdout(dev, y, groups, 0.15, 7)
    parts = [set(groups[index]) for index in (train, val, test)]
    assert all(not a & b for i, a in enumerate(parts) for b in parts[i + 1:])
    assert len(train) + len(val) + len(test) == len(y)


def test_heldout_labels_do_not_select_the_mlp_checkpoint():
    data = load_data()
    X, y = arrays(data, "condition", "poetry")
    dev, test = holdout(np.arange(len(y)), y, None, 0.15, 7)
    train, val = holdout(dev, y, None, 0.15, 7)
    flipped = y.copy()
    flipped[test] = 1 - flipped[test]
    config = replace(MLPConfig.from_settings(Runner().runs["a11"]["mlp"]), epochs=2)
    for before, after in zip(fit_predict(X, y, train, val, test, seed=7, config=config),
                             fit_predict(X, flipped, train, val, test, seed=7, config=config), strict=True):
        np.testing.assert_array_equal(before, after)
