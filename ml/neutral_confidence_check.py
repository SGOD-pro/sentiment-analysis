"""
Confidence check on neutral-class misclassifications from your DistilBERT
finetune. Answers one question: when the model gets 'neutral' wrong, is it
genuinely uncertain (probabilities close together, e.g. 0.45 vs 0.40) or
confidently wrong (0.85 vs 0.05)?

Close probabilities  -> model already "knows" these are boundary cases,
                         it's being forced into a hard decision. Ordinal
                         regression / soft-boundary reframing is justified.

Confidently wrong     -> this isn't a framing problem, the model is making
                         real errors it's sure about. That points back to
                         label noise or a genuine vocabulary/feature gap,
                         not a task-formulation issue. Don't build the
                         ordinal model on that evidence - go back to
                         targeted error analysis instead.

Run this AFTER your DistilBERT training script, reusing the same trainer/
test_ds objects if still in memory, or reload the saved model below.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

ROOT = "/content/drive/MyDrive/Dataset/embeddings_output"
MODEL_PATH = "./distilbert-sentiment-v2"  # adjust if your best checkpoint saved elsewhere

df = pd.read_parquet(f"{ROOT}/bge_clean_metadata.parquet")
test_idx = np.load(f"{ROOT}/clean_test_idx_v2.npy")

le = LabelEncoder().fit(["negative", "neutral", "positive"])
df["label_id"] = le.transform(df["label"])
test_df = df.iloc[test_idx][["id", "category", "text", "label_id"]].reset_index(drop=True)

NEUTRAL_ID = le.transform(["neutral"])[0]
LABEL_NAMES = list(le.classes_)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# batch inference, get full probability distributions not just argmax
all_probs = []
BATCH = 64
texts = test_df["text"].tolist()

with torch.no_grad():
    for i in range(0, len(texts), BATCH):
        batch_texts = texts[i:i + BATCH]
        enc = tokenizer(batch_texts, truncation=True, padding=True,
                         max_length=256, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        if i % (BATCH * 50) == 0:
            print(f"{i}/{len(texts)}")

all_probs = np.vstack(all_probs)  # shape (n_test, 3)
preds = all_probs.argmax(axis=1)

test_df["pred_label_id"] = preds
test_df["prob_negative"] = all_probs[:, le.transform(["negative"])[0]]
test_df["prob_neutral"] = all_probs[:, NEUTRAL_ID]
test_df["prob_positive"] = all_probs[:, le.transform(["positive"])[0]]
test_df["top_prob"] = all_probs.max(axis=1)
# margin between top and second predicted class - the core signal we want
sorted_probs = np.sort(all_probs, axis=1)
test_df["confidence_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]

# isolate the cases we actually care about: TRUE label is neutral, but
# model predicted something else
neutral_misses = test_df[
    (test_df["label_id"] == NEUTRAL_ID) & (test_df["pred_label_id"] != NEUTRAL_ID)
].copy()

print(f"\nTotal neutral misclassifications: {len(neutral_misses)}")
print(f"\nConfidence margin distribution on neutral misses:")
print(neutral_misses["confidence_margin"].describe())

# bucket into "genuinely uncertain" vs "confidently wrong"
UNCERTAIN_THRESHOLD = 0.15  # margin under this = model was close to a toss-up
confidently_wrong = (neutral_misses["confidence_margin"] > UNCERTAIN_THRESHOLD).sum()
genuinely_uncertain = (neutral_misses["confidence_margin"] <= UNCERTAIN_THRESHOLD).sum()

print(f"\nGenuinely uncertain (margin <= {UNCERTAIN_THRESHOLD}): "
      f"{genuinely_uncertain} ({genuinely_uncertain/len(neutral_misses)*100:.1f}%)")
print(f"Confidently wrong (margin > {UNCERTAIN_THRESHOLD}): "
      f"{confidently_wrong} ({confidently_wrong/len(neutral_misses)*100:.1f}%)")

print("\n" + "=" * 60)
if genuinely_uncertain / len(neutral_misses) > 0.6:
    print("MAJORITY genuinely uncertain -> ordinal/soft-boundary reframing")
    print("is justified. The model already 'knows' these are close calls.")
else:
    print("MAJORITY confidently wrong -> this is NOT a framing problem.")
    print("Reformulating to ordinal regression likely won't help. Go back")
    print("to per-row error inspection on the confidently-wrong cases -")
    print("this points to label noise or a real feature/vocabulary gap.")
print("=" * 60)

# save the confidently-wrong ones specifically for manual review - these are
# the cases that DON'T fit the "boundary ambiguity" story and need a look
confidently_wrong_df = neutral_misses[
    neutral_misses["confidence_margin"] > UNCERTAIN_THRESHOLD
].sort_values("confidence_margin", ascending=False)

confidently_wrong_df[[
    "id", "category", "text", "prob_negative", "prob_neutral", "prob_positive",
    "confidence_margin"
]].to_csv(f"{ROOT}/confidently_wrong_neutral_misses.csv", index=False)

print(f"\nSaved {len(confidently_wrong_df)} confidently-wrong cases to "
      f"confidently_wrong_neutral_misses.csv for manual review.")
print("Read the top 20-30 by confidence_margin (highest first) - these are")
print("your model's most confident mistakes. If they read as obviously")
print("neutral to you as a human, that's a real feature gap. If they read")
print("as obviously negative/positive to you too, that's likely a label error.")
