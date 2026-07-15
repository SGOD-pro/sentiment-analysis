"""
Your last learning curve (5k -> 48k rows, heuristic labels) was flat -
0.45 point gain for 9.6x the data. That's evidence you're not data-limited,
you're label-quality-limited (or model-capacity-limited, see model_swap_test.py).

This script tests the actual useful question: does a smaller amount of
MANUALLY VERIFIED data outperform a much larger amount of raw heuristic
data, at matched or even unfavorable sample sizes? If yes - and I'd bet on
yes given what you've already seen - that tells you to keep investing in
correction quality over raw volume, and gives you a real answer to "how
much data do I need": less than you think, if it's clean.

Also runs a per-category sufficiency check: at what per-category sample
size does adding more rows for that SPECIFIC category stop helping? This
matters because your categories aren't equally hard (Software/Pet_Supplies
sit ~8 points below Musical_Instruments) - a global answer to "how much
data" ignores that some categories may need more, others less.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score

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

# split train pool into "manually corrected" vs "raw heuristic label, never touched"
is_corrected = train_df["id"].isin(corrected_ids).values
corrected_pool_idx = train_idx[is_corrected]
raw_pool_idx = train_idx[~is_corrected]

print(f"Corrected pool available: {len(corrected_pool_idx)}")
print(f"Raw heuristic pool available: {len(raw_pool_idx)}")


def eval_at_size(pool_idx, size, seed=42):
    if size > len(pool_idx):
        return None
    rng = np.random.RandomState(seed)
    sample = rng.choice(pool_idx, size=size, replace=False)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight="balanced")
    clf.fit(X[sample], y[sample])
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    neutral_f1 = f1_score(y_test, preds, average=None, labels=[le.transform(["neutral"])[0]])[0]
    return {"train_size": size, "accuracy": acc, "macro_f1": macro_f1, "neutral_f1": neutral_f1}


# ---------------------------------------------------------------------------
# TEST 1: quality vs quantity, matched sample sizes where possible
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 1: Corrected-only data vs Raw heuristic data, matched sizes")
print("=" * 60)

max_corrected = len(corrected_pool_idx)
sizes_to_test = [s for s in [2000, 4000, max_corrected] if s <= max_corrected]

results = []
for size in sizes_to_test:
    r_corrected = eval_at_size(corrected_pool_idx, size)
    r_corrected["data_type"] = "corrected_only"
    results.append(r_corrected)

    r_raw = eval_at_size(raw_pool_idx, size)
    r_raw["data_type"] = "raw_heuristic"
    results.append(r_raw)

    # also test raw at a much LARGER size, to see how much raw volume it
    # takes to match corrected-data performance at the smaller size
    r_raw_10x = eval_at_size(raw_pool_idx, min(size * 10, len(raw_pool_idx)))
    if r_raw_10x:
        r_raw_10x["data_type"] = "raw_heuristic_10x_size"
        results.append(r_raw_10x)

comparison_df = pd.DataFrame(results)
print(comparison_df.to_string(index=False))
print("\nIf 'corrected_only' beats 'raw_heuristic' at the SAME size, and even")
print("beats 'raw_heuristic_10x_size', that's a direct, quantified answer:")
print("data quality is worth roughly [X]x more than raw volume for this task.")


# ---------------------------------------------------------------------------
# TEST 2: per-category sufficiency - where does more data stop helping,
# broken out per category instead of globally
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 2: Per-category learning curve (worst 5 categories only,")
print("full run across all 34 is expensive - start with the ones that matter)")
print("=" * 60)

WORST_CATEGORIES = [
    "Software", "Pet_Supplies", "Gift_Cards",
    "Subscription_Boxes", "Magazine_Subscriptions",
]

category_results = []
for cat in WORST_CATEGORIES:
    cat_train_idx = train_idx[train_df["category"].values == cat]
    cat_test_mask = df.iloc[test_idx]["category"].values == cat
    cat_test_idx = test_idx[cat_test_mask]

    if len(cat_train_idx) < 200 or len(cat_test_idx) < 50:
        print(f"{cat}: insufficient rows to test reliably, skipping")
        continue

    X_cat_test, y_cat_test = X[cat_test_idx], y[cat_test_idx]

    for frac in [0.25, 0.5, 0.75, 1.0]:
        size = int(len(cat_train_idx) * frac)
        if size < 50:
            continue
        rng = np.random.RandomState(42)
        sample = rng.choice(cat_train_idx, size=size, replace=False)
        clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight="balanced")
        clf.fit(X[sample], y[sample])
        preds = clf.predict(X_cat_test)
        f1 = f1_score(y_cat_test, preds, average="macro")
        category_results.append({"category": cat, "train_size": size, "macro_f1": f1})

cat_df = pd.DataFrame(category_results)
print(cat_df.to_string(index=False))
print("\nFor each category: if f1 is still climbing at 100% of available")
print("training rows, that category is genuinely data-starved and would")
print("benefit from more labeling. If it plateaus by 50-75%, more raw rows")
print("for that category won't help - the ceiling is elsewhere (label")
print("quality or model capacity).")
