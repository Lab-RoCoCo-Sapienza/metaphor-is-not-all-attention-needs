"""Execute a run from the settings file:  python -m probes <run> [settings.yaml]"""
import os
import sys

# Before the numerical libraries are imported: single-threaded maths, no display.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"


def main(arguments):
    from .runner import CONFIG, Runner

    runner = Runner(arguments[1] if len(arguments) == 2 else CONFIG)
    if len(arguments) not in (1, 2):
        raise SystemExit(f"{__doc__}\n\nRuns: all, " + ", ".join(runner.runs))
    if arguments[0] != "all" and arguments[0] not in runner.runs:
        raise SystemExit(f"Unknown run {arguments[0]!r}. Runs: all, " + ", ".join(runner.runs))
    runner.run(arguments[0])


if __name__ == "__main__":
    main(sys.argv[1:])
