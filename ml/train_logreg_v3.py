"""
First real signal after the merge: 1,122 relabeled + 1,461 model_wrong rows
identified. Trained on v3 split - fully leak-free, verified 0 overlap.
Compare directly to your pre-correction baselines below.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score

ROOT = "/content/drive/MyDrive/Dataset/embeddings_output"

df = pd.read_parquet(f"{ROOT}/bge_clean_metadata.parquet")
X = np.load(f"{ROOT}/bge_clean_embeddings.npy", mmap_mode="r")
train_idx = np.load(f"{ROOT}/clean_train_idx_v3.npy")
test_idx = np.load(f"{ROOT}/clean_test_idx_v3.npy")
registry = pd.read_csv(f"{ROOT}/audit_registry.csv")

# leak check, every script, no exceptions
reviewed_ids = set(registry["id"])
test_ids = set(df.iloc[test_idx]["id"])
leaked = reviewed_ids & test_ids
assert len(leaked) == 0, f"LEAK: {len(leaked)} reviewed ids in test set. STOP."
print("Leak check passed.\n")

le = LabelEncoder().fit(["negative", "neutral", "positive"])
y = le.transform(df["label"])

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

print(classification_report(y_test, preds, target_names=list(le.classes_), digits=4))

macro_f1 = f1_score(y_test, preds, average="macro")
neutral_f1 = f1_score(y_test, preds, average=None, labels=[le.transform(["neutral"])[0]])[0]
acc = accuracy_score(y_test, preds)

print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Neutral F1: {neutral_f1:.4f}")

print("\nCompare to pre-correction baselines (all leak-free, v2 split):")
print("  LogisticRegression: Macro F1 0.7808 | Neutral F1 0.6827")
print("  MLP:                Macro F1 0.7860 | Neutral F1 0.6884")
print("  DistilBERT finetune: Macro F1 0.7851 | Neutral F1 0.6901")
print("\nNote: train/test set composition changed (v3 excludes 9,990 reviewed")
print("ids from test entirely, vs v2 excluding 6,600) - test set size and")
print("makeup differs slightly from v2, so treat this as directionally")
print("comparable, not a perfectly controlled A/B. If neutral F1 moved by")
print("more than ~1 point, that's likely real signal, not just split noise.")
