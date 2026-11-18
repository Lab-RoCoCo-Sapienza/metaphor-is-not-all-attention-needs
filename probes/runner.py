"""The one place that reads settings and writes results.

An experiment module supplies the science and nothing else:

    SETTINGS                      the setting names it requires
    compute(settings, data, ...)  fit and return its scores; touches no files
    INPUTS         (optional)     settings that name another run; that run's scores are passed in
    TABLE, table_rows(scores)  (optional)  printed table to refill, and its numbers row by row
    render(scores, settings, folder)  (optional)  draw figures into the folder
    summary(scores)            (optional)  one line for the terminal
"""
from copy import deepcopy
from importlib import import_module
from pathlib import Path

import yaml

from .data import ROOT, load_data
from .tables import OUTPUT, write_table

CONFIG = Path(__file__).resolve().parent / "configs/paper.yaml"


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML silently keeps the last of two equal keys. A settings file must not."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                raise ValueError(f"Duplicate key {key!r} on line {key_node.start_mark.line + 1}")
            seen.add(key)
        return super().construct_mapping(node, deep)


def expect_keys(settings, keys, owner):
    """Every setting is required and nothing else is accepted."""
    missing, unknown = sorted(set(keys) - set(settings)), sorted(set(settings) - set(keys))
    if missing or unknown:
        raise ValueError(f"Settings of {owner}: missing {missing}, unknown {unknown}")


class Runner:
    def __init__(self, config=CONFIG):
        document = yaml.load(Path(config).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        expect_keys(document, {"mlp", "all", "runs"}, config)
        self.runs, self.default = document["runs"], document["all"]
        self._data = None
        self._scores = {}

    def settings(self, name):
        """A private copy of one run's settings, checked against its experiment."""
        if name not in self.runs:
            raise KeyError(f"Unknown run {name!r}. Available: all, {', '.join(self.runs)}")
        settings = deepcopy(self.runs[name])
        module = import_module(f"probes.experiments.{settings['experiment']}")
        expect_keys(settings, module.SETTINGS | {"experiment"}, f"run {name!r}")
        return settings, module

    def scores(self, name):
        """Compute a run at most once per process, after the runs it depends on."""
        if name not in self._scores:
            settings, module = self.settings(name)
            inputs = {key: self.scores(settings[key]) for key in getattr(module, "INPUTS", ())}
            if self._data is None:
                self._data = load_data()
            self._scores[name] = module.compute(settings, self._data, **inputs)
        return self._scores[name]

    def run(self, name):
        """Execute a run (or every run listed under `all`) and write what the paper reports."""
        if name == "all":
            for each in self.default:
                self.run(each)
            return
        settings, module = self.settings(name)
        scores = self.scores(name)
        if hasattr(module, "table_rows"):
            written = write_table(module.TABLE, module.table_rows(scores))
            print(written.relative_to(ROOT), flush=True)
        if hasattr(module, "render"):
            folder = OUTPUT / "figures"
            folder.mkdir(parents=True, exist_ok=True)
            module.render(scores, settings, folder)
            print(f"{name}: figures written to {folder.relative_to(ROOT)}", flush=True)
        if hasattr(module, "summary"):
            print(module.summary(scores), flush=True)

