"""
Build 72-dimensional attention feature vectors for linear probe training.

For each (sample_id, condition) pair with attentions.json, response.json and span
annotations, constructs a feature vector:

  1. For each generated token t and layer l:
       - Stack all heads, max-pool to obtain a per-key-position scalar (key_len,)
       - Average max-pooled attention over the prompt positions belonging to each
         functional group to obtain values[t, l, group]

  2. Average over layers within each of 4 layer clusters
  3. Average over generated tokens within each of 3 generation phases

  Feature shape: 4 clusters x 3 phases x 6 groups = 72 dimensions
  Feature names: c{cluster}_p{phase}_{GROUP}
  A feature is 0 when the prompt has no token of that group.

Usage (from the repository root):

    python build_features.py --raw_input <mlcommons_csv_dir> --split train
    python build_features.py --raw_input data/input/example.jsonl --split validation \\
        --annotations_dir data/input/example_spans

Output: CSV, one row per (sample_id, condition), in the format of data/probing/features.csv.
Columns: sample_id, condition, safety_label, verified, persona, c0_p0_FIGURATIVE, ...
"""

import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd


# Qwen3 chat-template tokens around the prompt text (enable_thinking=False):
# "<|im_start|>user\nQuestion:" before it,
# "Answer:<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" after it.
PREFIX_LEN = 5
SUFFIX_LEN = 11
N_LAYERS = 40            # Qwen3-14B
MAX_NEW_TOKENS = 50      # must match extract_attentions.py

# Ward clustering of layers on the calibration set (paper, Appendix A.6)
LAYER_CLUSTERS = [
    [0],
    list(range(1, 5)),
    list(range(5, 7)),
    list(range(7, 40)),
]
N_CLUSTERS = len(LAYER_CLUSTERS)  # 4

PHASE_SIZES = [17, 17, 16]
assert sum(PHASE_SIZES) == MAX_NEW_TOKENS
PHASE_RANGES = [list(range(sum(PHASE_SIZES[:i]), sum(PHASE_SIZES[:i + 1])))
                for i in range(len(PHASE_SIZES))]
N_PHASES = len(PHASE_RANGES)  # 3

GROUPS    = sorted(["FIGURATIVE", "FUNCTION_WORD", "HARMFUL_PAYLOAD",
                    "PUNCTUATION", "SETUP", "TECHNICAL"])
N_GROUPS  = len(GROUPS)  # 6
GROUP_IDX = {g: i for i, g in enumerate(GROUPS)}

CONDITIONS = ["poetry", "prose"]

FEATURE_COLS = [
    f"c{c}_p{p}_{g}"
    for c in range(N_CLUSTERS)
    for p in range(N_PHASES)
    for g in GROUPS
]  # 72 names, ordered (cluster, phase, group)


def build_feature_vector(
    attn_data: dict,
    token_roles: list,
    input_len: int,
) -> np.ndarray:
    """
    Return 72-dim float32 feature vector for one (sample, condition) pair.

    attn_data   : parsed attentions.json
    token_roles : list[list[str]] from span_annotations, length = n_content_tokens
    input_len   : total input tokens for this sample
    """
    n_content = input_len - PREFIX_LEN - SUFFIX_LEN
    if len(token_roles) != n_content:
        raise ValueError(
            f"token_roles length {len(token_roles)} != expected {n_content} "
            f"(input_len={input_len})"
        )

    # Map each group to its absolute key positions within the input
    group_positions: dict[str, list[int]] = defaultdict(list)
    for i, roles in enumerate(token_roles):
        pos = PREFIX_LEN + i
        for role in roles:
            if role in GROUP_IDX:
                group_positions[role].append(pos)

    values = np.zeros((MAX_NEW_TOKENS, N_LAYERS, N_GROUPS), dtype=np.float32)

    for t in range(MAX_NEW_TOKENS):
        layers_raw = attn_data[str(t)]["layers"]
        # (N_LAYERS, 1, n_heads, 1, key_len)
        all_layers = np.array(layers_raw, dtype=np.float32)
        # (N_LAYERS, key_len)
        max_attn = all_layers[:, 0, :, 0, :].max(axis=1)

        for role, positions in group_positions.items():
            g_idx = GROUP_IDX[role]
            # (N_LAYERS,)
            values[t, :, g_idx] = max_attn[:, positions].mean(axis=1)

    # (N_CLUSTERS, N_PHASES, N_GROUPS)
    features = np.zeros((N_CLUSTERS, N_PHASES, N_GROUPS), dtype=np.float32)
    for c, cl_layers in enumerate(LAYER_CLUSTERS):
        for p, ph_tokens in enumerate(PHASE_RANGES):
            subset = values[np.ix_(ph_tokens, cl_layers)]  # (n_phase, n_cluster, N_GROUPS)
            features[c, p, :] = subset.mean(axis=(0, 1))

    return features.flatten()  # (72,)


def get_safety_label(response: dict) -> Optional[str]:
    """Human-verified safety label of the response, or None if it has not been verified.

    Only verified labels (final_safety_status) are used, as in the paper. The label in
    safety_status comes from the mlcommons CSV run and disagrees with the verified label
    on about 20% of the main dataset, so it is never used as a fallback.
    """
    return response.get("final_safety_status") if response.get("verified") else None


def read_personas(raw_input: str) -> dict:
    """Map (sample_id, condition) to the attacker persona, from the extraction input.

    raw_input is the same input as for extract_attentions.py: an mlcommons CSV
    directory or a JSONL file with one poem/prose pair per line.
    """
    personas = {}
    path = Path(raw_input)
    if path.is_dir():
        for fname, condition in [("baseline_fail.csv", "prose"), ("baseline_success.csv", "prose"),
                                 ("poetic_fail.csv", "poetry"), ("poetic_success.csv", "poetry")]:
            if not (path / fname).exists():
                continue
            with open(path / fname, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sample_id = row.get("prompt_id", "").split("::", 1)[-1]
                    personas[(sample_id, condition)] = row.get("persona") or None
    else:
        with open(path) as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    for condition in CONDITIONS:
                        personas[(sample.get("release_prompt_id"), condition)] = sample.get("persona")
    return personas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_input", required=True,
        help="Input of the extraction (mlcommons CSV directory or JSONL file), "
             "used for the attacker persona.")
    parser.add_argument(
        "--results_dir", default="extraction_results/Qwen3-14B",
        help="Model results directory (contains train/ and/or validation/ subdirs).")
    parser.add_argument(
        "--annotations_dir", default="extraction_results/token_attention_patterns",
        help="Base directory of the span annotations: <annotations_dir>/<split>/span_annotations/<annotator>/.")
    parser.add_argument(
        "--split", default="train", choices=["train", "validation"],
        help="Which split to process (default: train).")
    parser.add_argument(
        "--annotator", default="gpt-5.4",
        help="Annotator subdirectory under span_annotations/ (default: gpt-5.4, used in the paper).")
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path. Defaults to <results_dir>/features/features_<split>.csv.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse rows from <output>.ckpt.jsonl and process only the missing pairs.")
    args = parser.parse_args()

    split       = args.split
    results_dir = Path(args.results_dir) / split
    ann_dir     = Path(args.annotations_dir) / split / "span_annotations" / args.annotator
    out_path    = Path(args.output) if args.output else Path(args.results_dir) / "features" / f"features_{split}.csv"
    ckpt_path   = out_path.with_suffix(".ckpt.jsonl")

    if not results_dir.is_dir():
        parser.error(f"results directory not found: {results_dir}")
    if not ann_dir.is_dir():
        parser.error(f"annotation directory not found: {ann_dir}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    personas = read_personas(args.raw_input)

    print(f"Split:       {split}")
    print(f"Results:     {results_dir}")
    print(f"Annotations: {ann_dir}")
    print(f"Output:      {out_path}")
    print(f"Checkpoint:  {ckpt_path}")

    # Reuse the checkpoint only with --resume; otherwise start fresh
    ckpt_rows: list[dict] = []
    done_pairs: set[tuple[str, str]] = set()
    if ckpt_path.exists() and not args.resume:
        ckpt_path.unlink()
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    ckpt_rows.append(row)
                    done_pairs.add((row["sample_id"], row["condition"]))
        print(f"Resuming:    {len(ckpt_rows)} rows already done\n")

    sample_dirs = sorted(results_dir.glob("*"))
    total = sum(
        1 for sd in sample_dirs for cond in CONDITIONS
        if (sd / cond / "attentions.json").exists()
    )
    print(f"Total pairs: {total}  |  already done: {len(done_pairs)}")
    print("=" * 60)

    new_rows: list[dict] = []
    n_skipped = 0
    n_seen = 0

    with open(ckpt_path, "a") as ckpt_file:
        for sd in sample_dirs:
            for cond in CONDITIONS:
                attn_path = sd / cond / "attentions.json"
                if not attn_path.exists():
                    continue

                n_seen += 1

                if (sd.name, cond) in done_pairs:
                    continue

                resp_path = sd / cond / "response.json"
                ann_path  = ann_dir / f"{sd.name}_{cond}_spans.json"

                missing = [p.name for p in [resp_path, ann_path] if not p.exists()]
                if missing:
                    print(f"  [{n_seen}/{total}] {sd.name}/{cond} skipping, missing: {missing}")
                    n_skipped += 1
                    continue

                print(f"  [{n_seen}/{total}] {sd.name}/{cond}", end=" ", flush=True)

                try:
                    with open(attn_path) as f: attn_data = json.load(f)
                    with open(resp_path) as f: resp      = json.load(f)
                    with open(ann_path)  as f: ann       = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"CORRUPT JSON ({e}) — skipping")
                    n_skipped += 1
                    continue

                try:
                    fvec = build_feature_vector(
                        attn_data,
                        ann["token_roles"],
                        resp["input_len"],
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    n_skipped += 1
                    continue

                row = {
                    "sample_id":    sd.name,
                    "condition":    cond,
                    "safety_label": get_safety_label(resp),
                    "verified":     bool(resp.get("verified", False)),
                    "persona":      personas.get((sd.name, cond)),
                }
                for col, val in zip(FEATURE_COLS, fvec):
                    row[col] = float(val)

                ckpt_file.write(json.dumps(row) + "\n")
                ckpt_file.flush()
                new_rows.append(row)
                print("ok")

    all_rows = ckpt_rows + new_rows
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print(f"Saved {len(df)} rows × {len(FEATURE_COLS)} features → {out_path}")
    print(f"Skipped: {n_skipped}")
    print(f"\nCondition:\n{df['condition'].value_counts().to_string()}")
    print(f"\nSafety label:\n{df['safety_label'].value_counts(dropna=False).to_string()}")
    print(f"\nPersona:\n{df['persona'].value_counts(dropna=False).to_string()}")
    print(f"\nVerified: {df['verified'].sum()} / {len(df)}")
