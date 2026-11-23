
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from common.encoding import encode_state, INPUT_SIZE

# File locations
INPUT_PATH = Path("data/processed/decisions.jsonl")
OUTPUT_DIR = Path("data/processed")

# Load the parsed poker decisions
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

print("Total decisions:", len(records))

# Convert each state into 108 numerical features
X = np.stack([encode_state(record) for record in records])

# The correct action for each state
y = np.array(
    [record["action_id"] for record in records],
    dtype=np.int64
)

# Identify which hand each decision came from
groups = np.array(
    [record["hand_id"] for record in records]
)

# Check the data
assert X.shape[1] == INPUT_SIZE
assert len(X) == len(y)
assert np.isfinite(X).all()

print("Feature dimensions:", X.shape[1])

# Split 70% training, 30% remaining
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    splitter.split(X, y, groups)
)

# Split remaining hands into validation and test
splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_local, test_local = next(
    splitter2.split(
        X[temp_idx],
        y[temp_idx],
        groups[temp_idx]
    )
)

val_idx = temp_idx[val_local]
test_idx = temp_idx[test_local]

# Confirm no poker hand appears in multiple sets
assert not (
    set(groups[train_idx]) & set(groups[val_idx])
)
assert not (
    set(groups[train_idx]) & set(groups[test_idx])
)
assert not (
    set(groups[val_idx]) & set(groups[test_idx])
)

# Save the datasets
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for name, idx in [
    ("train", train_idx),
    ("val", val_idx),
    ("test", test_idx)
]:
    np.save(OUTPUT_DIR / f"X_{name}.npy", X[idx])
    np.save(OUTPUT_DIR / f"y_{name}.npy", y[idx])

    print(f"{name}: {len(idx)} examples")

print("\nDataset preparation complete!")