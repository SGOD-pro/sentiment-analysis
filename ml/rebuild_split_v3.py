"""
v3 split rebuild. Excludes EVERY id that's ever been manually reviewed
(original 6,600 correction rounds + the 3,390 confidence-audit rows) from
the test pool - not just the ones whose label changed. You looked directly
at all 3,390, so all of them are compromised as blind test data, whether
you relabeled them or confirmed them as-is.

Do not reuse clean_test_idx_v2 for anything after the merge. That file is
contaminated. This produces v3, the new source of truth going forward.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path("/content/drive/MyDrive/Dataset/embeddings_output")
DATASET_PATH = ROOT / "bge_clean_metadata.parquet"
AUDIT_REGISTRY_PATH = ROOT / "audit_registry.csv"

df = pd.read_parquet(DATASET_PATH)
registry = pd.read_csv(AUDIT_REGISTRY_PATH)

reviewed_ids = set(registry["id"])
print(f"Total ids ever manually reviewed (all rounds + confidence audit): {len(reviewed_ids)}")

le = LabelEncoder().fit(["negative", "neutral", "positive"])
y_all = le.transform(df["label"])

eligible_mask = ~df["id"].isin(reviewed_ids)
eligible_indices = np.where(eligible_mask)[0]
reviewed_indices = np.where(~eligible_mask)[0]

print(f"Eligible (never reviewed) pool: {len(eligible_indices)}")
print(f"Reviewed pool (goes to train only): {len(reviewed_indices)}")

# split ONLY among untouched rows - guaranteed clean test set
clean_train_idx, clean_test_idx = train_test_split(
    eligible_indices,
    test_size=0.2,
    random_state=123,  # new seed, new lineage - don't reuse old seeds across versions
    stratify=y_all[eligible_indices],
)

# ALL reviewed ids go into train, never test
clean_train_idx = np.concatenate([clean_train_idx, reviewed_indices])

np.save(ROOT / "clean_train_idx_v3.npy", clean_train_idx)
np.save(ROOT / "clean_test_idx_v3.npy", clean_test_idx)

print(f"\nv3 train: {len(clean_train_idx)} ({len(clean_train_idx)/len(df)*100:.1f}%)")
print(f"v3 test:  {len(clean_test_idx)} ({len(clean_test_idx)/len(df)*100:.1f}%)")

# verify stratification held
for name, idx in [("train", clean_train_idx), ("test", clean_test_idx)]:
    dist = pd.Series(y_all[idx]).value_counts(normalize=True).sort_index()
    print(f"\n{name} label distribution:\n{dist}")

# mandatory leak check - must be 0
test_ids = set(df.iloc[clean_test_idx]["id"])
leaked = reviewed_ids & test_ids
print(f"\nLeak check: {len(leaked)} reviewed ids in v3 test set.")
assert len(leaked) == 0, "LEAK in v3 rebuild - stop, do not train on this split."
print("Clean. Safe to train on clean_train_idx_v3 / clean_test_idx_v3.")
