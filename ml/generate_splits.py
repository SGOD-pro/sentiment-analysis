"""
Generates train/val/test splits (stratified, index-tracked) for both BGE
and MiniLM embeddings, plus a manual-review sample for editing labels.

CRITICAL: the manual-review sample is pulled from TRAIN ONLY. Never review
or hand-edit labels in val/test - if you do, you contaminate your own
evaluation and any accuracy improvement you see afterward is fake, not real.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path("/content/drive/MyDrive/Datasets/Sentiment/embeddings_output")

df = pd.read_parquet(ROOT / "amazon_reviews.parquet")
bge_x = np.load(ROOT / "bge_embeddings.npy")
mini_x = np.load(ROOT / "mini_embeddings.npy")

assert len(df) == len(bge_x) == len(mini_x), (
    f"Row mismatch: df={len(df)}, bge={len(bge_x)}, mini={len(mini_x)}. "
    f"Fix this before splitting anything - misaligned rows here poison "
    f"every downstream result silently."
)

y = df["label"].values
indices = np.arange(len(df))

# --- split 1: carve out test (15%) first, stratified ---
train_val_idx, test_idx = train_test_split(
    indices,
    test_size=0.15,
    random_state=42,
    stratify=y,
)

# --- split 2: carve val (15% of ORIGINAL total) out of remaining 85% ---
# 0.15 / 0.85 = ~0.176 to get 15% of the original total, not 15% of the remainder
val_fraction_of_remainder = 0.15 / 0.85
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=val_fraction_of_remainder,
    random_state=42,
    stratify=y[train_val_idx],
)

print(f"Train: {len(train_idx)} ({len(train_idx)/len(df)*100:.1f}%)")
print(f"Val:   {len(val_idx)} ({len(val_idx)/len(df)*100:.1f}%)")
print(f"Test:  {len(test_idx)} ({len(test_idx)/len(df)*100:.1f}%)")

# verify stratification actually held - don't just trust the parameter, check it
for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    dist = pd.Series(y[idx]).value_counts(normalize=True).sort_index()
    print(f"\n{name} label distribution:\n{dist}")


def save_split(name: str, idx: np.ndarray):
    split_df = df.iloc[idx].copy()
    split_df.to_parquet(ROOT / f"{name}.parquet", index=False)
    np.save(ROOT / f"{name}_bge_embeddings.npy", bge_x[idx])
    np.save(ROOT / f"{name}_mini_embeddings.npy", mini_x[idx])
    print(f"Saved {name}: {len(idx)} rows -> "
          f"{name}.parquet, {name}_bge_embeddings.npy, {name}_mini_embeddings.npy")


save_split("train", train_idx)
save_split("val", val_idx)
save_split("test", test_idx)

# ---------------------------------------------------------------------------
# Manual review sample - PULLED FROM TRAIN ONLY. Do not change this to pull
# from val or test. Editing labels there = contaminating your own eval.
# ---------------------------------------------------------------------------
N_REVIEW = 250  # adjust within your stated 200-300 range

review_sample_idx = np.random.RandomState(42).choice(
    train_idx, size=min(N_REVIEW, len(train_idx)), replace=False
)

review_df = df.iloc[review_sample_idx][["id", "category", "text", "label"]].copy()
review_df = review_df.rename(columns={"label": "original_label"})
review_df["corrected_label"] = review_df["original_label"]  # you edit this column
review_df["notes"] = ""  # optional free-text column for why you changed it

review_path_csv = ROOT / "manual_review_sample.csv"
review_path_parquet = ROOT / "manual_review_sample.parquet"
review_df.to_csv(review_path_csv, index=False)
review_df.to_parquet(review_path_parquet, index=False)

print(f"\nManual review sample: {len(review_df)} rows from TRAIN ONLY")
print(f"Saved to: {review_path_csv}")
print(f"Edit 'corrected_label' column in Excel/Sheets, then re-import to")
print(f"apply fixes back into your train set before retraining.")
