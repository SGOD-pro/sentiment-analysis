"""
Fast manual label-correction loop.

CRITICAL RULE, DO NOT VIOLATE THIS ACROSS ANY ROUND: sample for review comes
from TRAIN indices only, never from idx_test. If you correct test-set labels
based on model predictions and retrain, your test accuracy becomes fake -
you're no longer measuring generalization, you're measuring how well the
model matches labels you personally nudged toward its own prior output.
Test set must stay 100% frozen and untouched across all 4-5 rounds for the
accuracy comparison between rounds to mean anything.

Workflow per round:
  1. get_review_sample()   -> random N ids from TRAIN, with text + current label
  2. you manually read them, build corrections = {id: correct_label, ...}
  3. apply_corrections()   -> pushes fixes into the original df + saves it
  4. retrain on the corrected train set
  5. repeat, EXCLUDING already-reviewed ids so you're not reviewing the same
     rows every round
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Datasets/Sentiment/embeddings_output")
DF_PATH = ROOT / "amazon_reviews.parquet"
REVIEWED_LOG_PATH = ROOT / "reviewed_ids_log.csv"  # tracks what's already been checked

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def get_review_sample(train_idx: np.ndarray, df: pd.DataFrame, n: int = 200, seed: int = None):
    """
    Pulls a random sample from TRAIN indices only.
    seed=None -> different rows every round (recommended, so you're not
    stuck reviewing the same 200 rows 5 times). Pass an int if you want
    reproducibility for some reason.

    Excludes ids already reviewed in previous rounds, tracked via the log file.
    """
    if REVIEWED_LOG_PATH.exists():
        reviewed_ids = set(pd.read_csv(REVIEWED_LOG_PATH)["id"].tolist())
    else:
        reviewed_ids = set()

    train_df = df.iloc[train_idx]
    train_df = train_df[~train_df["id"].isin(reviewed_ids)]

    if len(train_df) < n:
        print(f"WARNING: only {len(train_df)} unreviewed train rows left, "
              f"returning all of them instead of {n}.")
        n = len(train_df)

    rng = np.random.RandomState(seed) if seed is not None else np.random
    sample_idx = rng.choice(train_df.index.values, size=n, replace=False)

    sample = df.loc[sample_idx, ["id", "category", "text", "label"]].copy()
    sample["current_label_name"] = sample["label"].map(LABEL_NAMES)

    print(f"Sample of {len(sample)} rows from TRAIN only (test set untouched).")
    print("Read 'text' + 'current_label_name', then build your corrections dict:")
    print('  corrections = {134: 0, 5821: 2, ...}   # id -> correct label INT')
    return sample


def apply_corrections(df: pd.DataFrame, corrections: dict, save_path: Path = DF_PATH):
    """
    corrections: {id: correct_label_int}
    Pushes fixes into df by id match (not positional index - safer), logs
    which ids were reviewed this round (whether changed or confirmed correct),
    and saves.

    Pass ALL reviewed ids in `corrections` even if you didn't change them -
    e.g. {134: 0, 5821: 2, 9012: 9012_original_label} - so the log correctly
    excludes them from future rounds. If you only log the ones you changed,
    you'll keep re-reviewing rows you already confirmed were fine.
    """
    ids_to_fix = list(corrections.keys())
    mask = df["id"].isin(ids_to_fix)

    before = df.loc[mask, ["id", "label"]].copy()

    df.loc[mask, "label"] = df.loc[mask, "id"].map(corrections)

    changed = before.merge(
        df.loc[mask, ["id", "label"]], on="id", suffixes=("_before", "_after")
    )
    changed = changed[changed["label_before"] != changed["label_after"]]

    print(f"Reviewed: {len(ids_to_fix)} rows")
    print(f"Actually changed: {len(changed)} rows")
    if len(changed) > 0:
        print(changed)

    # log ALL reviewed ids (changed or not) so future rounds skip them
    log_entry = pd.DataFrame({"id": ids_to_fix})
    if REVIEWED_LOG_PATH.exists():
        existing_log = pd.read_csv(REVIEWED_LOG_PATH)
        log_entry = pd.concat([existing_log, log_entry]).drop_duplicates()
    log_entry.to_csv(REVIEWED_LOG_PATH, index=False)

    df.to_parquet(save_path, index=False)
    print(f"Saved corrected df to {save_path}")
    print(f"Total unique ids reviewed across all rounds so far: {len(log_entry)}")

    return df


# ---------------------------------------------------------------------------
# USAGE (run across cells in Colab, one round at a time):
#
# df = pd.read_parquet(DF_PATH)
# train_idx = ...  # reuse the SAME train_idx from your original split -
#                   # do not regenerate it with a new random_state, or you
#                   # risk pulling from what used to be val/test
#
# sample = get_review_sample(train_idx, df, n=200)
# sample.to_csv(ROOT / "round_1_to_review.csv", index=False)
# -> open in Excel/Sheets, read text + current_label_name, decide corrections
#
# corrections = {134: 0, 5821: 2, 9012: 1}   # build this dict from your review
# df = apply_corrections(df, corrections)
#
# -> retrain on corrected train set, evaluate on your UNTOUCHED test set,
#    compare test accuracy to before. THIS comparison is the one that's real,
#    because test labels never moved.
# ---------------------------------------------------------------------------
