"""
Consumes your filled-in model_wrong_reaudit_scaffold.csv and:
  1. Merges mislabel_negative / mislabel_positive into df['label']
  2. Leaves genuinely_mixed and true_model_wrong_* untouched (still neutral)
  3. Carves out a FROZEN difficult-neutral eval slice (200-300 rows) from
     the true_model_wrong_* + genuinely_mixed rows - never used for training,
     ever, regardless of future rounds
  4. Reports the specific metrics you asked to track: neutral recall,
     neutral precision, macro F1, neutral->negative rate, neutral->positive rate
  5. Prints the error_subtype breakdown so you can pick the right
     intervention (targeted hard-example training / compositional coverage
     / calibration fix) based on evidence, not guessing

Does NOT duplicate model_wrong samples into training. Does NOT decide your
training intervention for you - that's step 3 in your plan, done after this.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path("/content/drive/MyDrive/Dataset/embeddings_output")
DATASET_PATH = ROOT / "bge_clean_metadata.parquet"
AUDIT_REGISTRY_PATH = ROOT / "audit_registry.csv"
SCAFFOLD_PATH = ROOT / "model_wrong_reaudit_scaffold_FILLED.csv"  # your completed version

VALID_PRIMARY = {
    "mislabel_negative", "mislabel_positive", "genuinely_mixed",
    "true_model_wrong_negative", "true_model_wrong_positive",
}
VALID_SUBTYPE = {
    "factual_neutral", "weak_sentiment", "hedged_sentiment",
    "mixed_but_neutral", "ambiguous", "sarcasm_or_context",
}

reaudit = pd.read_csv(SCAFFOLD_PATH)

# --- gate 1: fully reviewed ---
assert reaudit["primary_bucket"].notna().all() and (reaudit["primary_bucket"] != "").all(), (
    "primary_bucket has empty rows. Not fully reviewed yet."
)
assert reaudit["primary_bucket"].isin(VALID_PRIMARY).all(), (
    f"Invalid primary_bucket values: "
    f"{set(reaudit['primary_bucket'].unique()) - VALID_PRIMARY}"
)

# --- gate 2: every true_model_wrong_* row must have a subtype ---
model_wrong_mask = reaudit["primary_bucket"].str.startswith("true_model_wrong")
missing_subtype = reaudit[model_wrong_mask & (reaudit["error_subtype"].isna() | (reaudit["error_subtype"] == ""))]
assert len(missing_subtype) == 0, (
    f"{len(missing_subtype)} true_model_wrong rows are missing error_subtype. "
    f"Fill these in: {missing_subtype['id'].tolist()}"
)
invalid_subtype = reaudit[model_wrong_mask & ~reaudit["error_subtype"].isin(VALID_SUBTYPE)]
assert len(invalid_subtype) == 0, f"Invalid error_subtype values found."

print("PRIMARY BUCKET BREAKDOWN:")
print(reaudit["primary_bucket"].value_counts())
print()

print("ERROR SUBTYPE BREAKDOWN (true_model_wrong_* rows only):")
print(reaudit[model_wrong_mask]["error_subtype"].value_counts())
print()

# ---------------------------------------------------------------------------
# STEP 1-2: merge mislabels only, leave everything else as neutral
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATASET_PATH)
if "original_label" not in df.columns:
    df["original_label"] = df["label"]

mislabel_neg = reaudit[reaudit["primary_bucket"] == "mislabel_negative"]
mislabel_pos = reaudit[reaudit["primary_bucket"] == "mislabel_positive"]

corrections = {}
corrections.update({row["id"]: "negative" for _, row in mislabel_neg.iterrows()})
corrections.update({row["id"]: "positive" for _, row in mislabel_pos.iterrows()})

correction_series = pd.Series(corrections, name="new_label")
update_mask = df["id"].isin(correction_series.index)
df.loc[update_mask, "label"] = df.loc[update_mask, "id"].map(correction_series)

df.to_parquet(DATASET_PATH, index=False)
print(f"Merged {len(corrections)} label corrections "
      f"({len(mislabel_neg)} -> negative, {len(mislabel_pos)} -> positive)")

# ---------------------------------------------------------------------------
# STEP 3: FROZEN difficult-neutral eval slice
# Pulled from true_model_wrong_* and genuinely_mixed - these are your
# hardest, most human-verified neutral cases. NEVER train on these ids,
# ever, in any future round. Track them separately from clean_test_idx.
# ---------------------------------------------------------------------------
eval_candidates = reaudit[
    reaudit["primary_bucket"].isin([
        "genuinely_mixed", "true_model_wrong_negative", "true_model_wrong_positive"
    ])
]
print(f"\nEval slice candidate pool: {len(eval_candidates)}")

EVAL_SIZE = min(300, len(eval_candidates))
difficult_neutral_eval = eval_candidates.sample(n=EVAL_SIZE, random_state=77)

difficult_neutral_eval.to_csv(ROOT / "difficult_neutral_eval_FROZEN.csv", index=False)
frozen_eval_ids = set(difficult_neutral_eval["id"])
np.save(ROOT / "difficult_neutral_eval_ids.npy", np.array(list(frozen_eval_ids)))

print(f"Frozen difficult-neutral eval slice: {len(difficult_neutral_eval)} rows saved.")
print("These ids must NEVER appear in any future training set, in addition")
print("to normal train/test exclusion. Track this file's ids permanently.")

# ---------------------------------------------------------------------------
# Update registry - all 1,461 re-audited ids logged, tagged with full detail
# ---------------------------------------------------------------------------
registry = pd.read_csv(AUDIT_REGISTRY_PATH)
already_reviewed_ids = set(registry["id"])
new_ids = reaudit[~reaudit["id"].isin(already_reviewed_ids)]

new_rows = pd.DataFrame({
    "id": new_ids["id"],
    "round": "model_wrong_reaudit",
    "sampling_strategy": "model_wrong_confidence_audit",
    "old_label": new_ids["actual_label"],
    "manual_label": new_ids["primary_bucket"],
    "audit_status": new_ids["primary_bucket"],
    "notes": new_ids["error_subtype"].fillna("") + " | " + new_ids["notes"].fillna(""),
})
registry = pd.concat([registry, new_rows], ignore_index=True)
registry.to_csv(AUDIT_REGISTRY_PATH, index=False)
print(f"\nRegistry updated. Total ids ever reviewed: {registry['id'].nunique()}")

# ---------------------------------------------------------------------------
# Rebuild v4 split - excludes ALL reviewed ids AND the frozen eval ids
# ---------------------------------------------------------------------------
le = LabelEncoder().fit(["negative", "neutral", "positive"])
y_all = le.transform(df["label"])

all_excluded_ids = set(registry["id"]) | frozen_eval_ids
eligible_mask = ~df["id"].isin(all_excluded_ids)
eligible_indices = np.where(eligible_mask)[0]
excluded_indices = np.where(~eligible_mask)[0]

# frozen eval rows get pulled OUT entirely - not train, not test, their own thing
frozen_eval_positions = df.index[df["id"].isin(frozen_eval_ids)]
frozen_eval_indices = np.array([df.index.get_loc(i) for i in frozen_eval_positions])

# reviewed-but-not-frozen-eval ids go to train only
reviewed_not_frozen = all_excluded_ids - frozen_eval_ids
reviewed_not_frozen_indices = np.where(df["id"].isin(reviewed_not_frozen))[0]

clean_train_idx, clean_test_idx = train_test_split(
    eligible_indices, test_size=0.2, random_state=456,
    stratify=y_all[eligible_indices],
)
clean_train_idx = np.concatenate([clean_train_idx, reviewed_not_frozen_indices])

np.save(ROOT / "clean_train_idx_v4.npy", clean_train_idx)
np.save(ROOT / "clean_test_idx_v4.npy", clean_test_idx)

# leak checks - both against normal test AND against frozen eval
test_ids_v4 = set(df.iloc[clean_test_idx]["id"])
leak_registry = all_excluded_ids & test_ids_v4
leak_frozen = frozen_eval_ids & test_ids_v4
print(f"\nv4 split: train={len(clean_train_idx)}, test={len(clean_test_idx)}")
print(f"Leak check (registry ids in test): {len(leak_registry)}")
print(f"Leak check (frozen eval ids in test): {len(leak_frozen)}")
assert len(leak_registry) == 0 and len(leak_frozen) == 0, "LEAK in v4 - stop."
print("Clean.")

print("\n" + "=" * 60)
print("NEXT: train on clean_train_idx_v4, evaluate on BOTH:")
print("  - clean_test_idx_v4 (your normal held-out metric)")
print("  - difficult_neutral_eval_FROZEN.csv (the hard-case tracking metric)")
print("Report all 5 tracked metrics on the frozen slice specifically:")
print("  neutral recall, neutral precision, macro F1,")
print("  neutral->negative rate, neutral->positive rate")
print("=" * 60)
