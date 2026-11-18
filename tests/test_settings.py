"""The settings file is complete, strict, and written without repetition."""
import pytest

from probes.models.mlp import MLPConfig
from probes.runner import Runner, expect_keys


def test_every_run_has_exactly_the_settings_its_experiment_requires():
    runner = Runner()
    for name in runner.runs:
        settings, _ = runner.settings(name)
        for block in (settings.get("mlp"), settings.get("mlp_evaluation", {}).get("model")):
            if block is not None:
                MLPConfig.from_settings(block)
    assert set(runner.default) <= set(runner.runs)


def test_shared_values_are_written_once():
    runs = Runner().runs
    assert runs["a8"]["mlp_evaluation"] == runs["table1"]["mlp_evaluation"]
    assert runs["a8"]["prose_draw_seeds"] == runs["table1"]["prose_draw_seeds"]
    table1_mlp, persona_mlp = runs["table1"]["mlp_evaluation"]["model"], runs["a11"]["mlp"]
    assert {key for key in table1_mlp if table1_mlp[key] != persona_mlp[key]} == {"standardize"}


def test_unknown_missing_and_repeated_settings_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        expect_keys({"seed": 1, "sead": 2}, {"seed"}, "a run")
    with pytest.raises(ValueError, match="missing"):
        expect_keys({}, {"seed"}, "a run")
    with pytest.raises(TypeError):
        MLPConfig.from_settings({**Runner().runs["a11"]["mlp"], "droput": 0.5})
    path = tmp_path / "paper.yaml"
    path.write_text("mlp: {}\nall: []\nruns:\n  a:\n    seed: 1\n    seed: 2\n")
    with pytest.raises(ValueError, match="Duplicate key"):
        Runner(path)
    with pytest.raises(KeyError, match="Unknown run"):
        Runner().settings("unknown_run")


def test_settings_handed_to_an_experiment_cannot_change_the_file_in_memory():
    runner = Runner()
    settings, _ = runner.settings("table1")
    settings["mlp_evaluation"]["model"]["dropout"] = 0.9
    assert runner.settings("a8")[0]["mlp_evaluation"]["model"]["dropout"] == 0.1
