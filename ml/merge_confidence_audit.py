"""
Merges YOUR manually reviewed corrections from
confidently_wrong_neutral_misses_labeled.csv back into the original dataset.

Schema: manual_label = bucket ('mislabel' / 'genuinely_mixed' / 'model_wrong')
        corrected_label = the actual sentiment value to write into df['label']

Only 'mislabel' rows should change the ground-truth label. 'genuinely_mixed'
and 'model_wrong' rows keep corrected_label='neutral' (no change) - they get
logged in the registry as CONFIRMED, not corrected, and 'model_wrong' rows
get flagged separately since they indicate a model calibration issue, not
a data issue - don't let them get lost in a generic 'reviewed' bucket.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Dataset/embeddings_output")
DATASET_PATH = ROOT / "bge_clean_metadata.parquet"
AUDIT_REGISTRY_PATH = ROOT / "audit_registry.csv"
REVIEWED_PATH = ROOT / "confidently_wrong_neutral_misses_labeled.csv"

VALID_BUCKETS = {"mislabel", "genuinely_mixed", "model_wrong"}
VALID_LABELS = {"negative", "neutral", "positive"}

reviewed = pd.read_csv(REVIEWED_PATH)

# --- gate 1: fully filled, valid buckets, valid corrected labels ---
assert reviewed["manual_label"].notna().all(), "manual_label has empty rows - not fully reviewed."
assert reviewed["manual_label"].isin(VALID_BUCKETS).all(), (
    f"Invalid bucket values found: {set(reviewed['manual_label'].unique()) - VALID_BUCKETS}"
)
assert reviewed["corrected_label"].isin(VALID_LABELS).all(), (
    f"Invalid corrected_label values found: "
    f"{set(reviewed['corrected_label'].unique()) - VALID_LABELS}"
)

# --- gate 2: internal consistency check - genuinely_mixed and model_wrong
# should NOT have changed the label away from neutral. If they did, that's
# a labeling contradiction (why bucket it as "leave as neutral" but then
# assign a different corrected_label?) - catch it, don't silently merge it ---
inconsistent = reviewed[
    (reviewed["manual_label"].isin(["genuinely_mixed", "model_wrong"])) &
    (reviewed["corrected_label"] != "neutral")
]
assert len(inconsistent) == 0, (
    f"{len(inconsistent)} rows bucketed as genuinely_mixed/model_wrong but "
    f"corrected_label isn't 'neutral' - contradiction, fix these rows:\n"
    f"{inconsistent[['id', 'manual_label', 'corrected_label']]}"
)

print("Bucket breakdown:")
print(reviewed["manual_label"].value_counts())
print()

actual_changes = reviewed[reviewed["manual_label"] == "mislabel"]
print(f"Actual label changes to merge: {len(actual_changes)}")
print(actual_changes["corrected_label"].value_counts())

model_wrong_flagged = reviewed[reviewed["manual_label"] == "model_wrong"]
print(f"\nFlagged as model calibration issue (label NOT changed, worth its own")
print(f"investigation separately): {len(model_wrong_flagged)}")

# ---------------------------------------------------------------------------
# MERGE into original dataset - only mislabel rows actually change df['label']
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATASET_PATH)

if "original_label" not in df.columns:
    df["original_label"] = df["label"]  # preserve pre-correction ground truth, always

correction_map = actual_changes.set_index("id")["corrected_label"]
update_mask = df["id"].isin(correction_map.index)

df.loc[update_mask, "label"] = df.loc[update_mask, "id"].map(correction_map)

df.to_parquet(DATASET_PATH, index=False)
print(f"\nSaved. Updated {update_mask.sum()} rows in {DATASET_PATH}")

# ---------------------------------------------------------------------------
# Update audit registry - ALL 3390 reviewed ids get logged (even the ones
# where the label didn't change), tagged with their bucket so you can filter
# 'model_wrong' cases later for a dedicated calibration investigation
# ---------------------------------------------------------------------------
if AUDIT_REGISTRY_PATH.exists():
    registry = pd.read_csv(AUDIT_REGISTRY_PATH)
else:
    registry = pd.DataFrame(columns=[
        "id", "round", "sampling_strategy", "old_label", "manual_label",
        "audit_status", "notes"
    ])

already_reviewed_ids = set(registry["id"]) if len(registry) > 0 else set()
new_ids = reviewed[~reviewed["id"].isin(already_reviewed_ids)]

new_registry_rows = pd.DataFrame({
    "id": new_ids["id"],
    "round": "confidence_audit",
    "sampling_strategy": "confidently_wrong_neutral",
    "old_label": new_ids["actual_label"],
    "manual_label": new_ids["corrected_label"],
    "audit_status": new_ids["manual_label"],  # preserves the bucket: mislabel/genuinely_mixed/model_wrong
    "notes": "",
})

registry = pd.concat([registry, new_registry_rows], ignore_index=True)
registry.to_csv(AUDIT_REGISTRY_PATH, index=False)
print(f"Registry updated. Total ids ever reviewed: {registry['id'].nunique()}")

# save the model_wrong subset separately - this is your next investigation,
# not something to fold into the general audit trail and forget about
model_wrong_flagged.to_csv(ROOT / "model_wrong_calibration_issues.csv", index=False)
print(f"\nSaved {len(model_wrong_flagged)} model_wrong rows to "
      f"model_wrong_calibration_issues.csv for a separate follow-up.")

# ---------------------------------------------------------------------------
# CRITICAL: these 3390 ids came from your TEST set (clean_test_idx_v2).
# You MUST rebuild train/test split before training or evaluating again -
# do not skip this, the old clean_test_idx_v2 is now contaminated.
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("STOP - REQUIRED NEXT STEP")
print("="*60)
print("These 3390 ids were sampled from your TEST set. clean_test_idx_v2.npy")
print("is now contaminated for 1,122 of them (the ones whose label changed)")
print("- and arguably all 3390 should move to train, since you've now looked")
print("at every one of them directly. Rebuild the split (v3) BEFORE running")
print("any training or evaluation script again.")


# ---------------------------------------------------------------------------
# CRITICAL: verify no leak happened in this merge
# ---------------------------------------------------------------------------
test_idx = np.load(ROOT / "clean_test_idx_v2.npy")
test_ids = set(df.iloc[test_idx]["id"])
corrected_ids_now = set(registry["id"])
leaked = corrected_ids_now & test_ids
print(f"\nLeak check after this merge: {len(leaked)} corrected ids in test set.")
if len(leaked) > 0:
    print("STOP - some corrected ids landed in your test set. This happens if")
    print("confidently_wrong_neutral_misses.csv was built from BOTH train and")
    print("test rows rather than train-only. Do not train/evaluate again until")
    print("you've confirmed which split these ids came from and rebuilt")
    print("clean_test_idx accordingly.")
