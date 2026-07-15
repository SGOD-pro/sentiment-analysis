"""
Trains on v4, evaluates on both the standard test set AND the frozen
difficult-neutral slice, reporting exactly the metrics you specified:
neutral recall, neutral precision, macro F1, neutral->negative rate,
neutral->positive rate.

Run this AFTER reaudit_merge_and_freeze_eval.py has produced v4 files.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

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


def report_metrics(X_eval, y_true, label):
    y_pred = clf.predict(X_eval)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    neutral_recall = recall_score(y_true, y_pred, labels=[NEUTRAL_ID], average="macro")
    neutral_precision = precision_score(y_true, y_pred, labels=[NEUTRAL_ID], average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[NEGATIVE_ID, NEUTRAL_ID, POSITIVE_ID])
    # cm rows = true, cols = pred. neutral row is index 1.
    neutral_row = cm[1]
    neutral_total = neutral_row.sum()
    neutral_to_negative_rate = neutral_row[0] / neutral_total if neutral_total > 0 else float("nan")
    neutral_to_positive_rate = neutral_row[2] / neutral_total if neutral_total > 0 else float("nan")

    print(f"\n{'='*55}\n{label}\n{'='*55}")
    print(f"Macro F1:              {macro_f1:.4f}")
    print(f"Neutral recall:        {neutral_recall:.4f}")
    print(f"Neutral precision:     {neutral_precision:.4f}")
    print(f"Neutral -> negative:   {neutral_to_negative_rate:.4f}")
    print(f"Neutral -> positive:   {neutral_to_positive_rate:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred, order neg/neu/pos):\n{cm}")

    return {
        "eval_set": label, "macro_f1": macro_f1,
        "neutral_recall": neutral_recall, "neutral_precision": neutral_precision,
        "neutral_to_negative_rate": neutral_to_negative_rate,
        "neutral_to_positive_rate": neutral_to_positive_rate,
    }


results = []

# standard test set
results.append(report_metrics(X[test_idx], y[test_idx], "Standard test set (v4)"))

# frozen difficult-neutral slice - the real signal
frozen_indices = df.index[df["id"].isin(frozen_eval["id"])].values
frozen_positions = np.array([df.index.get_loc(i) for i in frozen_indices])
results.append(report_metrics(
    X[frozen_positions], y[frozen_positions], "FROZEN difficult-neutral slice"
))

summary = pd.DataFrame(results)
print(f"\n{'='*55}\nSUMMARY\n{'='*55}")
print(summary.to_string(index=False))

print("\nThe frozen-slice numbers are your real signal on whether corrections")
print("are helping the HARD cases specifically, not just the easy majority")
print("that a standard random test set is dominated by. Track this table")
print("across every future round - it should be the metric that decides if")
print("an intervention worked, not the standard test set macro F1 alone.")
