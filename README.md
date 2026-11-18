## Metaphor Is Not All Attention Needs


This repository allows for the reproduction of the results of *Metaphor Is Not All Attention Needs*, a mechanistic interpretability paper on adversarial poetry, with a focus on attention's patterns. 

The paper was accepted at NeurIPS 2026, and the pre-print can be read on [arXiv](https://arxiv.org/abs/2605.12128
).

The code currently in the repository refits the probing results from the released 72-dimensional feature vectors and regenerates the paper’s tables.

To run the code simply:

```sh
uv sync --locked
uv run python -m probes all
```

`all` refits Table 1 and A.11, writes both tables to `results/output/tables/`.

| Paper item | Output |
|---|---|
| Table 1 | `results/output/tables/table1.tex` |
| A.11 persona table | `results/output/tables/a11.tex` |
| A.7 safety coefficient charts | `results/output/figures/` |

## Settings

[probes/configs/paper.yaml](probes/configs/paper.yaml) holds the configuration parameters of the experiments.  

## Layout

```
data/                    features.csv (2,395 rows) and exclusions.csv (46 prompt IDs)
probes/
  configs/paper.yaml     every setting of every run
  runner.py              reads the settings, calls an experiment, writes the table or figures
  experiments/           one module per paper item; each computes scores and touches no files
  models/                the three probes: linear.py (logistic regression), svc.py, mlp.py
  data.py  sampling.py   the cleaned pool, balanced draws and holdouts
  tables.py  charts.py   refill a printed table; draw a coefficient chart
results/
  paper/tables/          original manuscript tables
  paper/figures/         the three original coefficient plots
  output/                written by the runs above
tests/
```

## Experiments


- **Table 1.** Logistic regression (L1) and the RBF SVC use five outer folds, grouped by prompt for the format probe, with C chosen by inner cross-validation after scaling the outer training fold. The MLP is scored on five seeded 72/13/15 partitions. Prose safety uses five balanced subsamples of 122 safe and 122 unsafe prompts.
- **A.11.** Five persona contrasts on ungrouped folds. 
- **A.8.** Table 1’s MLP evaluation repeated over six hidden-layer shapes and two dropout rates.

