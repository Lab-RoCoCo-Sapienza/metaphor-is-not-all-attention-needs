"""The ordered feature matrix and the eligibility filter shared by all experiments."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CATEGORIES = ("FIGURATIVE", "FUNCTION_WORD", "HARMFUL_PAYLOAD", "PUNCTUATION", "SETUP", "TECHNICAL")
FEATURES = [f"c{c}_p{p}_{group}" for c in range(4) for p in range(3) for group in CATEGORIES]


def load_data():
    frame = pd.read_csv(DATA / "features.csv", float_precision="round_trip")
    if frame.duplicated(["sample_id", "condition"]).any():
        raise ValueError("Duplicate (sample_id, condition) key")
    if frame[FEATURES].isna().any().any() or frame["persona"].isna().any():
        raise ValueError("Missing feature or persona")
    exclusions = pd.read_csv(DATA / "exclusions.csv")
    return frame.loc[~frame.sample_id.isin(exclusions.sample_id)].reset_index(drop=True)


def arrays(frame, target, positive, dtype=np.float32):
    return frame[FEATURES].to_numpy(dtype=dtype), (frame[target] == positive).to_numpy(dtype=int)
