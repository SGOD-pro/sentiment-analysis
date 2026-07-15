"""
Asymmetric confidence rule, built around your stated priority: missing a
genuinely-negative review (false neutral) is worse than a false positive/
negative on a genuinely-neutral review.

The SYMMETRIC threshold from before pushes ANY uncertain prediction toward
neutral - including uncertain NEGATIVE predictions, which is exactly the
error you just said is costliest. This version only softens uncertain
POSITIVE predictions toward neutral. Negative predictions are trusted at
a lower confidence bar, since under-flagging negative reviews is your
stated worse outcome.

Re-run the sweep with this asymmetric rule and compare against the
symmetric baseline - the metric that matters now is different: track
NEGATIVE recall as closely as neutral recall, since that's the class
you said you can't afford to under-predict.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, confusion_matrix

ROOT = "/content/drive/MyDrive/Dataset/embeddings_output"

df = pd.read_parquet(f"{ROOT}/bge_clean_metadata.parquet")
X = np.load(f"{ROOT}/bge_clean_embeddings.npy", mmap_mode="r")
train_idx = np.load(f"{ROOT}/clean_train_idx_v4.npy")
test_idx = np.load(f"{ROOT}/clean_test_idx_v4.npy")
frozen_eval = pd.read_csv(f"{ROOT}/difficult_neutral_eval_FROZEN.csv")

le = LabelEncoder().fit(["negative", "neutral", "positive"])
y = le.transform(df["label"])
NEGATIVE_ID = le.transform(["negative"])[0]
NEUTRAL_ID = le.transform(["neutral"])[0]
POSITIVE_ID = le.transform(["positive"])[0]

clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight="balanced")
clf.fit(X[train_idx], y[train_idx])

X_test, y_test = X[test_idx], y[test_idx]
frozen_indices = df.index[df["id"].isin(frozen_eval["id"])].values
frozen_positions = np.array([df.index.get_loc(i) for i in frozen_indices])
X_frozen, y_frozen = X[frozen_positions], y[frozen_positions]

test_probs = clf.predict_proba(X_test)
frozen_probs = clf.predict_proba(X_frozen)


def apply_asymmetric_rule(probs, positive_margin_threshold, negative_margin_threshold=0.0):
    """
    Only softens toward neutral when the top prediction is POSITIVE and
    uncertain. Negative predictions get a much lower bar (default 0.0 =
    never overridden) since under-flagging negative reviews is costlier
    than an occasional wrong negative call on a neutral review.
    """
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    argmax_preds = probs.argmax(axis=1)

    final_preds = argmax_preds.copy()

    # only reassign to neutral when: predicted positive AND margin below threshold
    positive_uncertain = (argmax_preds == POSITIVE_ID) & (margins < positive_margin_threshold)
    final_preds[positive_uncertain] = NEUTRAL_ID

    # negative predictions: only reassign if margin is below the (much
    # stricter, default-zero) negative threshold - protects negative recall
    negative_uncertain = (argmax_preds == NEGATIVE_ID) & (margins < negative_margin_threshold)
    final_preds[negative_uncertain] = NEUTRAL_ID

    return final_preds


def report(y_true, preds, label, include_macro=True):
    print(f"\n{'='*55}\n{label}\n{'='*55}")
    neg_recall = recall_score(y_true, preds, labels=[NEGATIVE_ID], average="macro")
    neutral_recall = recall_score(y_true, preds, labels=[NEUTRAL_ID], average="macro")
    print(f"Negative recall: {neg_recall:.4f}  <- your priority metric, watch this doesn't drop")
    print(f"Neutral recall:  {neutral_recall:.4f}")
    if include_macro:
        macro_f1 = f1_score(y_true, preds, average="macro")
        print(f"Macro F1: {macro_f1:.4f}")
    return neg_recall, neutral_recall


print("BASELINE (plain argmax)")
baseline_preds_test = test_probs.argmax(axis=1)
baseline_preds_frozen = frozen_probs.argmax(axis=1)
report(y_test, baseline_preds_test, "Standard test - baseline")
report(y_frozen, baseline_preds_frozen, "Frozen slice - baseline", include_macro=False)

print("\n\n" + "#"*60)
print("# ASYMMETRIC RULE SWEEP - positive_margin_threshold varies,")
print("# negative predictions always trusted (negative_margin_threshold=0.0)")
print("#"*60)

results = []
for pos_threshold in [0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
    test_preds = apply_asymmetric_rule(test_probs, pos_threshold, negative_margin_threshold=0.0)
    frozen_preds = apply_asymmetric_rule(frozen_probs, pos_threshold, negative_margin_threshold=0.0)

    test_neg_recall = recall_score(y_test, test_preds, labels=[NEGATIVE_ID], average="macro")
    test_neutral_recall = recall_score(y_test, test_preds, labels=[NEUTRAL_ID], average="macro")
    test_macro_f1 = f1_score(y_test, test_preds, average="macro")
    frozen_neutral_recall = recall_score(y_frozen, frozen_preds, labels=[NEUTRAL_ID], average="macro")

    results.append({
        "pos_threshold": pos_threshold,
        "test_negative_recall": test_neg_recall,
        "test_neutral_recall": test_neutral_recall,
        "test_macro_f1": test_macro_f1,
        "frozen_neutral_recall": frozen_neutral_recall,
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\ntest_negative_recall should stay FLAT across all thresholds here -")
print("that's the point, we never touch negative predictions. If it moves,")
print("something's wrong with the rule logic.")
print("\nfrozen_neutral_recall will climb, but LESS than the symmetric version")
print("did, because we're only fixing the positive-leaning half of the")
print("miscalibration (160 of 1461 model_wrong cases were positive-predicted,")
print("1301 were negative-predicted - so this asymmetric rule only directly")
print("addresses the smaller slice of the problem, by design, to protect")
print("negative recall).")
print("\nThis is the real tradeoff of your stated priority: you get less")
print("frozen-recall improvement than the symmetric rule gave you, in")
print("exchange for never sacrificing negative recall. That's the correct")
print("trade GIVEN what you told me matters most - but be clear-eyed that")
print("it means the negative-predicted chunk of your model_wrong cases")
print("(1301 rows, the majority) stays largely unaddressed by this rule.")
print("Those need the retraining/label-smoothing route instead, not a")
print("threshold fix, if you want to improve them without risking negative recall.")
