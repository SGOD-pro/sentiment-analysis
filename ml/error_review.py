"""
Manual error-review dataframe.
Splits WITH index tracking + stratify, trains a baseline, and builds a
queryable df so you can actually look at what's wrong instead of trusting
a single aggregate number.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --- sanity check for leakage BEFORE trusting anything below ---
dupe_count = df.duplicated(subset="text").sum()
print(f"Duplicate review texts in dataset: {dupe_count} "
      f"({dupe_count/len(df)*100:.1f}%)")
if dupe_count / len(df) > 0.02:
    print("^ That's high enough to be inflating your accuracy via train/test "
          "leakage. Consider df.drop_duplicates(subset='text') before splitting, "
          "or better: split by product/category group, not by row.")
print()

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}  # confirm this matches
                                                              # your actual encoding
                                                              # before trusting output

X = bge_x
y = df["label"].values
indices = np.arange(len(df))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices,
    test_size=0.2,
    random_state=42,
    stratify=y,          # put this back - non-negotiable
)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=list(LABEL_NAMES.values())))

# --- build the reviewable dataframe, aligned back to original rows ---
results_df = df.iloc[idx_test].copy()
results_df["predicted_label"] = y_pred
results_df["actual_label_name"] = results_df["label"].map(LABEL_NAMES)
results_df["predicted_label_name"] = results_df["predicted_label"].map(LABEL_NAMES)
results_df["correct"] = results_df["label"] == results_df["predicted_label"]

# columns you actually want to eyeball
results_df = results_df[[
    "id", "category", "text", "actual_label_name", "predicted_label_name", "correct"
]]

# 200 random samples for manual review
review_sample = results_df.sample(n=200, random_state=42).reset_index(drop=True)

print(f"\nOverall test accuracy: {results_df['correct'].mean():.4f}")
print(f"Review sample built: {len(review_sample)} rows -> `review_sample`")

# Useful queries once you have this:
#   review_sample[~review_sample.correct]                          -> only errors
#   review_sample[review_sample.actual_label_name == "neutral"]    -> just neutral class
#   results_df.groupby("category")["correct"].mean().sort_values() -> worst categories first
#   results_df[~results_df.correct].groupby(
#       ["actual_label_name","predicted_label_name"]).size()       -> confusion pairs, full test set
