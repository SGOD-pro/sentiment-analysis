"""
Corrected-only underperformed matched-size raw data (0.7173 vs 0.7710 macro
F1) - almost certainly because your correction rounds oversampled hard,
boundary-ambiguous cases (registry defaults to "neutral_to_negative"
strategy), giving the model zero exposure to easy, clear-signal examples
during training. This tests whether COMBINING corrected + a broad random
raw sample beats either alone - hard-example correction should help MOST
when layered onto a training set that also has normal decision-boundary
coverage, not as a total replacement for it.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score

ROOT = "/content/drive/MyDrive/Dataset/embeddings_output"

df = pd.read_parquet(f"{ROOT}/bge_clean_metadata.parquet")
X = np.load(f"{ROOT}/bge_clean_embeddings.npy", mmap_mode="r")
train_idx = np.load(f"{ROOT}/clean_train_idx_v2.npy")
test_idx = np.load(f"{ROOT}/clean_test_idx_v2.npy")
registry = pd.read_csv(f"{ROOT}/audit_registry.csv")

le = LabelEncoder().fit(["negative", "neutral", "positive"])
y = le.transform(df["label"])
X_test, y_test = X[test_idx], y[test_idx]

corrected_ids = set(registry["id"])
train_df = df.iloc[train_idx]
is_corrected = train_df["id"].isin(corrected_ids).values
corrected_pool_idx = train_idx[is_corrected]
raw_pool_idx = train_idx[~is_corrected]

print(f"Corrected pool: {len(corrected_pool_idx)}")
print(f"Raw pool: {len(raw_pool_idx)}")


def train_eval(train_indices, label, seed=42):
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight="balanced")
    clf.fit(X[train_indices], y[train_indices])
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    neutral_f1 = f1_score(y_test, preds, average=None, labels=[le.transform(["neutral"])[0]])[0]

    print(f"\n{'='*55}\n{label} (n={len(train_indices)})\n{'='*55}")
    print(classification_report(y_test, preds, target_names=list(le.classes_), digits=4))
    print(f"Macro F1: {macro_f1:.4f} | Neutral F1: {neutral_f1:.4f}")

    return {"label": label, "n": len(train_indices), "accuracy": acc,
            "macro_f1": macro_f1, "neutral_f1": neutral_f1}


results = []

# baseline 1: corrected only (already ran, rerunning for a clean side-by-side table)
results.append(train_eval(corrected_pool_idx, "corrected_only"))

# baseline 2: raw only, matched to full raw pool size available up to a cap
# (using full pool here since that's your realistic "don't use corrections" baseline)
raw_full_sample = np.random.RandomState(42).choice(
    raw_pool_idx, size=min(len(raw_pool_idx), 60000), replace=False
)
results.append(train_eval(raw_full_sample, "raw_only_60k"))

# THE ACTUAL TEST: corrected + raw combined, at a few different raw:corrected ratios,
# so you can see if there's a sweet spot rather than just "more raw is always better"
for raw_size in [10000, 20000, 40000, 60000, len(raw_pool_idx)]:
    if raw_size > len(raw_pool_idx):
        continue
    raw_sample = np.random.RandomState(42).choice(raw_pool_idx, size=raw_size, replace=False)
    combined = np.concatenate([corrected_pool_idx, raw_sample])
    results.append(train_eval(combined, f"combined_corrected_plus_raw_{raw_size}"))

summary_df = pd.DataFrame(results)
print(f"\n{'='*55}\nSUMMARY\n{'='*55}")
print(summary_df.to_string(index=False))

best = summary_df.loc[summary_df["macro_f1"].idxmax()]
print(f"\nBest config: {best['label']} - macro F1 {best['macro_f1']:.4f}, "
      f"neutral F1 {best['neutral_f1']:.4f}")
print("\nIf a combined variant beats BOTH corrected_only and raw_only_60k,")
print("that confirms: correction data is valuable but only when layered onto")
print("broad coverage, not as a standalone training set. Use that combined")
print("config going forward - not corrected-only, not raw-only.")
print("\nIf macro F1 keeps climbing as raw_size increases even in the combined")
print("runs, you're still not data-saturated on the RAW side specifically -")
print("worth testing beyond your current raw pool size eventually.")
