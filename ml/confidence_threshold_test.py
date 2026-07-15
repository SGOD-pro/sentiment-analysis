"""
Two things:
1. Fixed frozen-slice eval - drops macro F1 and neutral precision on that
   slice (both are meaningless artifacts when true labels are 100% neutral),
   reports neutral recall as the honest headline number instead.
2. Confidence-threshold post-hoc rule: if the model's top prediction doesn't
   clear a margin over the second-place class, default to neutral instead
   of trusting argmax. No retraining - tests the calibration hypothesis in
   ~20 minutes using your EXISTING trained model's probabilities.

Run this before touching label smoothing or any retraining. If this alone
recovers a meaningful chunk of frozen-slice neutral recall, that's strong,
cheap evidence the problem is exactly what the taxonomy suggested:
overconfidence at the decision boundary, not missing representational
capacity.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

ROOT = "/content/drive/MyDrive/Dataset/embeddings_output"

df = pd.read_parquet(f"{ROOT}/bge_clean_metadata.parquet")
X = np.load(f"{ROOT}/bge_clean_embeddings.npy", mmap_mode="r")
train_idx = np.load(f"{ROOT}/clean_train_idx_v4.npy")
test_idx = np.load(f"{ROOT}/clean_test_idx_v4.npy")
frozen_eval = pd.read_csv(f"{ROOT}/difficult_neutral_eval_FROZEN.csv")

le = LabelEncoder().fit(["negative", "neutral", "positive"])
y = le.transform(df["label"])
NEUTRAL_ID = le.transform(["neutral"])[0]
NEGATIVE_ID = le.transform(["negative"])[0]
POSITIVE_ID = le.transform(["positive"])[0]

clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight="balanced")
clf.fit(X[train_idx], y[train_idx])

frozen_indices = df.index[df["id"].isin(frozen_eval["id"])].values
frozen_positions = np.array([df.index.get_loc(i) for i in frozen_indices])
X_frozen, y_frozen = X[frozen_positions], y[frozen_positions]

X_test, y_test = X[test_idx], y[test_idx]


def honest_report(X_eval, y_true, label, all_classes_present=True):
    """Only reports macro F1/precision if the slice actually has all 3
    classes present as true labels - otherwise those metrics are artifacts."""
    probs = clf.predict_proba(X_eval)
    preds = probs.argmax(axis=1)

    print(f"\n{'='*55}\n{label}\n{'='*55}")

    neutral_recall = recall_score(y_true, preds, labels=[NEUTRAL_ID], average="macro")
    print(f"Neutral recall: {neutral_recall:.4f}  <- the honest headline number")

    if all_classes_present:
        macro_f1 = f1_score(y_true, preds, average="macro")
        neutral_precision = precision_score(y_true, preds, labels=[NEUTRAL_ID], average="macro", zero_division=0)
        print(f"Macro F1:          {macro_f1:.4f}")
        print(f"Neutral precision: {neutral_precision:.4f}")
    else:
        print("(Macro F1 / neutral precision skipped - not meaningful when")
        print(" true labels are 100% one class, would be an artifact.)")

    cm = confusion_matrix(y_true, preds, labels=[NEGATIVE_ID, NEUTRAL_ID, POSITIVE_ID])
    print(f"Confusion matrix (rows=true, cols=pred, order neg/neu/pos):\n{cm}")

    return preds, probs, neutral_recall


# --- baseline: standard argmax, for comparison ---
print("BASELINE (plain argmax, no threshold rule)")
_, _, standard_test_recall = honest_report(X_test, y_test, "Standard test set")
_, frozen_probs, frozen_baseline_recall = honest_report(
    X_frozen, y_frozen, "Frozen difficult-neutral slice", all_classes_present=False
)

# ---------------------------------------------------------------------------
# CONFIDENCE-THRESHOLD POST-HOC RULE
# If top prediction doesn't beat 2nd place by at least `margin_threshold`,
# default to neutral instead of trusting argmax.
# ---------------------------------------------------------------------------
def apply_threshold_rule(probs, margin_threshold):
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    argmax_preds = probs.argmax(axis=1)
    thresholded_preds = np.where(margins < margin_threshold, NEUTRAL_ID, argmax_preds)
    return thresholded_preds, margins


print("\n\n" + "#" * 60)
print("# CONFIDENCE-THRESHOLD RULE - testing several thresholds")
print("#" * 60)

threshold_results = []
for threshold in [0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
    test_probs = clf.predict_proba(X_test)
    test_thresh_preds, _ = apply_threshold_rule(test_probs, threshold)
    frozen_thresh_preds, _ = apply_threshold_rule(frozen_probs, threshold)

    test_macro_f1 = f1_score(y_test, test_thresh_preds, average="macro")
    test_neutral_recall = recall_score(y_test, test_thresh_preds, labels=[NEUTRAL_ID], average="macro")
    frozen_neutral_recall = recall_score(y_frozen, frozen_thresh_preds, labels=[NEUTRAL_ID], average="macro")

    threshold_results.append({
        "threshold": threshold,
        "test_macro_f1": test_macro_f1,
        "test_neutral_recall": test_neutral_recall,
        "frozen_neutral_recall": frozen_neutral_recall,
    })

results_df = pd.DataFrame(threshold_results)
print(results_df.to_string(index=False))

print("\nWhat to look for:")
print("- frozen_neutral_recall should climb as threshold increases (more")
print("  low-margin predictions get reassigned to neutral, which IS the")
print("  true label for every row in this slice - so this should improve).")
print("- test_macro_f1 tells you the COST: at higher thresholds, you're also")
print("  forcing genuinely-confident-and-correct negative/positive predictions")
print("  on the standard test set into neutral incorrectly. Find where")
print("  frozen_neutral_recall gains stop being worth the test_macro_f1 cost.")
print("\nIf there's a threshold where frozen recall jumps meaningfully (e.g.")
print("0.29 -> 0.50+) while test_macro_f1 drops only slightly (e.g. <1-2")
print("points), that's a cheap, real fix - ship it as a post-hoc rule,")
print("no retraining needed. If frozen recall barely moves at any threshold,")
print("this isn't purely a margin/calibration issue and label smoothing or")
print("retraining is the next real step instead.")
