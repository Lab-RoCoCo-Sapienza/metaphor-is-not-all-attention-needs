"""Reuse the manuscript's typesetting, replacing every reported statistic with its fit."""
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from .data import ROOT

RESULTS = ROOT / "probing_results"
OUTPUT = RESULTS / "output"
# Every reported statistic is a decimal.
STATISTIC = re.compile(r"[-+]?\d+\.\d+")


def write_table(name, rows):
    """rows follows the typeset row order and supplies every numeric statistic on each row."""
    source = (RESULTS / "paper" / "tables" / f"{name}.tex").read_text(encoding="utf-8").splitlines(keepends=True)
    data_rows = [i for i, line in enumerate(source) if "&" in line and STATISTIC.search(line)]
    for i, values in zip(data_rows, rows, strict=True):
        replacements = dict(zip((m.start() for m in STATISTIC.finditer(source[i])), values, strict=True))
        source[i] = STATISTIC.sub(lambda m: replacements[m.start()], source[i])
    dest = OUTPUT / "tables" / f"{name}.tex"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=dest.parent, mode="w", encoding="utf-8", delete=False) as f:
        try:
            f.writelines(source)
            f.close()
            os.replace(f.name, dest)
        finally:
            Path(f.name).unlink(missing_ok=True)
    return dest
