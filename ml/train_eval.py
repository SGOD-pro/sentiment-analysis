"""
Train a baseline classifier on top of your saved embeddings and compare
BGE vs MiniLM head-to-head. Don't skip the comparison - that's the whole
point of embedding with two models in the first place.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

ROOT = "/content"  # change if reading from Drive

# --- CONFIG: you MUST set this ---
LABEL_COLUMN = "label"   # <-- CHANGE THIS. What column are you predicting?
                          # e.g. "rating", "sentiment", "label" - whatever
                          # your amazon_reviews.csv actually calls it.
                          # If it doesn't exist, this script fails loudly
                          # at the assert below, on purpose.

# --- load data + embeddings, verify alignment ---
df = pd.read_parquet(f"{ROOT}/amazon_reviews.parquet")
bge = np.load(f"{ROOT}/bge_embeddings.npy")
mini = np.load(f"{ROOT}/mini_embeddings.npy")

assert LABEL_COLUMN in df.columns, (
    f"'{LABEL_COLUMN}' not in dataframe columns: {list(df.columns)}. "
    f"Fix LABEL_COLUMN above before doing anything else."
)
assert len(df) == len(bge) == len(mini), (
    f"Row count mismatch: df={len(df)}, bge={len(bge)}, mini={len(mini)}. "
    f"Something got reordered or truncated somewhere - do not proceed "
    f"until these match, your labels and embeddings won't line up."
)

y = df[LABEL_COLUMN].values

# --- check class balance BEFORE you stratify and BEFORE you trust accuracy ---
print("Class distribution:")
print(df[LABEL_COLUMN].value_counts(normalize=True))
print()
print("If one class is >80% of the data, accuracy is a useless metric here.")
print("Look at the f1/precision/recall breakdown below instead, per class.\n")


def evaluate(X: np.ndarray, name: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,          # non-negotiable for imbalanced classes
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")

    print(f"=== {name} ===")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f}")
    print(classification_report(y_test, preds))
    print()

    return {"name": name, "accuracy": acc, "f1_macro": f1_macro}


results = [
    evaluate(bge, "BGE-small"),
    evaluate(mini, "MiniLM-L6"),
]

winner = max(results, key=lambda r: r["f1_macro"])
print(f"Winner by macro F1: {winner['name']} ({winner['f1_macro']:.4f})")
print("Use that one going forward. Don't keep both in production for no reason.")
