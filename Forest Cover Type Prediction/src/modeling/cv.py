"""
cv.py

# Goal
Central place for cross-validation configuration.

# Outputs
- StratifiedKFold object

# Steps
1) Define SEED constant.
2) Return StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
3) (Optional) Helper to print class distribution per fold for debugging.
"""

# SEED = 42
from __future__ import annotations
from sklearn.model_selection import StratifiedKFold

random_state = 42
n_splits = 5
shuffle = True


# Build StratifiedKFold setup
def make_stratified_kfold(
        n_splits: int = n_splits,
        shuffle: bool = shuffle,
        random_state: int = random_state
    ):
    return StratifiedKFold(
        n_splits = n_splits,
        shuffle = shuffle,
        random_state = random_state
    )