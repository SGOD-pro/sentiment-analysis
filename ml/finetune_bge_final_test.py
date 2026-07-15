"""
FINAL architecture test, round 2 - with early stopping fixed.

Round 1 result: train loss dropped every epoch (0.484->0.396) while val
loss ROSE every epoch (0.491->0.518) - classic overfitting, and the script
saved whatever came out at epoch 3 (the worst val loss) with no safeguard.
Result: best-ever standard test macro F1 (0.8087) but worst-ever frozen
slice recall (0.2400) - the model got better at confident pattern-matching,
worse at genuinely ambiguous cases.

This version adds early stopping + load_best_model_at_end, tracking macro
F1 during training (not just loss) so the final saved encoder comes from
whichever epoch actually generalized best, not just whichever ran last.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, recall_score
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset

ROOT = "/content/drive/MyDrive/Dataset/embeddings_output"

df = pd.read_parquet(f"{ROOT}/bge_clean_metadata.parquet")
train_idx = np.load(f"{ROOT}/clean_train_idx_v4.npy")
test_idx = np.load(f"{ROOT}/clean_test_idx_v4.npy")
registry = pd.read_csv(f"{ROOT}/audit_registry.csv")
frozen_eval = pd.read_csv(f"{ROOT}/difficult_neutral_eval_FROZEN.csv")

reviewed_ids = set(registry["id"])
frozen_ids = set(frozen_eval["id"])
test_ids = set(df.iloc[test_idx]["id"])
assert len((reviewed_ids | frozen_ids) & test_ids) == 0, "LEAK - STOP."
print("Leak check passed.\n")

le = LabelEncoder().fit(["negative", "neutral", "positive"])
df["label_id"] = le.transform(df["label"])
NEGATIVE_ID = le.transform(["negative"])[0]
NEUTRAL_ID = le.transform(["neutral"])[0]

MODEL_NAME = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class BGEForClassification(nn.Module):
    def __init__(self, model_name, n_classes=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


train_df = df.iloc[train_idx][["text", "label_id"]].reset_index(drop=True)
test_df = df.iloc[test_idx][["text", "label_id"]].reset_index(drop=True)

val_frac = 0.08
rng = np.random.RandomState(42)
val_positions = rng.choice(len(train_df), size=int(len(train_df) * val_frac), replace=False)
val_mask = np.zeros(len(train_df), dtype=bool)
val_mask[val_positions] = True
val_df = train_df[val_mask].reset_index(drop=True)
fit_df = train_df[~val_mask].reset_index(drop=True)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

fit_ds = Dataset.from_pandas(fit_df.rename(columns={"label_id": "labels"})).map(tokenize, batched=True)
val_ds = Dataset.from_pandas(val_df.rename(columns={"label_id": "labels"})).map(tokenize, batched=True)

model = BGEForClassification(MODEL_NAME)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "neutral_f1": f1_score(labels, preds, average=None, labels=[NEUTRAL_ID])[0],
    }


training_args = TrainingArguments(
    output_dir=f"{ROOT}/bge-finetuned-classification-v2",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,  # more epochs available, but early stopping will cut it short if it overfits again
    weight_decay=0.01,
    warmup_ratio=0.06,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,       # THE FIX - roll back to best checkpoint, not last
    metric_for_best_model="f1_macro",  # track F1, not just loss
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    logging_steps=200,
    save_total_limit=2,
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=fit_ds, eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # THE FIX - stop before it overfits further
)
print("Finetuning BGE-small with classification head (early stopping enabled)...\n")
trainer.train()

print(f"\nBest checkpoint selected by Trainer: epoch with highest val f1_macro")
print("(check the epoch-by-epoch table above - the LAST printed row is not")
print("necessarily what got kept, load_best_model_at_end already rolled back)")

# ---------------------------------------------------------------------------
# STEP 2: DISCARD the classifier, keep only the finetuned encoder
# ---------------------------------------------------------------------------
finetuned_encoder = model.encoder
finetuned_encoder.save_pretrained(f"{ROOT}/bge-finetuned-encoder-only-v2")
tokenizer.save_pretrained(f"{ROOT}/bge-finetuned-encoder-only-v2")
print("\nSaved finetuned encoder v2 (best checkpoint by val f1_macro, classifier head discarded).")

# ---------------------------------------------------------------------------
# STEP 3: re-embed everything with the finetuned encoder
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
finetuned_encoder.to(device)
finetuned_encoder.eval()

def embed_with_finetuned(texts, batch_size=64):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
            outputs = finetuned_encoder(**enc)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)
    return np.vstack(all_embeddings)

print("\nRe-embedding train/test/frozen sets with finetuned encoder...")
all_train_texts = df.iloc[train_idx]["text"].tolist()
all_test_texts = df.iloc[test_idx]["text"].tolist()
frozen_indices = df.index[df["id"].isin(frozen_eval["id"])].values
all_frozen_texts = df.loc[frozen_indices]["text"].tolist()

X_train_new = embed_with_finetuned(all_train_texts)
X_test_new = embed_with_finetuned(all_test_texts)
X_frozen_new = embed_with_finetuned(all_frozen_texts)

y_train = df.iloc[train_idx]["label_id"].values
y_test = df.iloc[test_idx]["label_id"].values
y_frozen = df.loc[frozen_indices]["label_id"].values

# ---------------------------------------------------------------------------
# STEP 4: fresh LogisticRegression on the NEW embeddings - directly
# comparable to your original frozen-BGE LR baseline
# ---------------------------------------------------------------------------
print("\nTraining LogisticRegression on finetuned embeddings...")
lr_new = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight="balanced")
lr_new.fit(X_train_new, y_train)

test_preds = lr_new.predict(X_test_new)
frozen_preds = lr_new.predict(X_frozen_new)

print("\n" + "="*55 + "\nFINETUNED-BGE + LR - Standard test\n" + "="*55)
print(classification_report(y_test, test_preds, target_names=list(le.classes_), digits=4))
new_macro_f1 = f1_score(y_test, test_preds, average="macro")
new_neg_recall = recall_score(y_test, test_preds, labels=[NEGATIVE_ID], average="macro")
new_neutral_recall = recall_score(y_test, test_preds, labels=[NEUTRAL_ID], average="macro")
print(f"Macro F1: {new_macro_f1:.4f} | Neg recall: {new_neg_recall:.4f} | Neutral recall: {new_neutral_recall:.4f}")

new_frozen_recall = recall_score(y_frozen, frozen_preds, labels=[NEUTRAL_ID], average="macro")
print(f"\nFrozen slice neutral recall: {new_frozen_recall:.4f}")

print("\n" + "="*55 + "\nCOMPARISON TO EXISTING RESULTS\n" + "="*55)
print("Frozen BGE + LR (v4 baseline):     neg_recall=0.7687, neutral_recall=0.7128, frozen_recall=0.2933")
print("Frozen BGE + MLP + threshold:      neg_recall=0.7915, neutral_recall=0.7426, frozen_recall=0.2967  <- current best")
print("DistilBERT finetune (v4):          neg_recall=0.7723, neutral_recall=0.7286, frozen_recall=0.2133")
print("Finetuned BGE round 1 (overfit):   neg_recall=0.7814, neutral_recall=0.7498, frozen_recall=0.2400")
print(f"Finetuned BGE round 2 (early stop): neg_recall={new_neg_recall:.4f}, neutral_recall={new_neutral_recall:.4f}, frozen_recall={new_frozen_recall:.4f}")
print("\nDecision criteria (unchanged from round 1): beats MLP+threshold by 2+")
print("points on neg_recall OR frozen_recall = worth the deployment complexity.")
print("If round 2 frozen_recall recovered toward/past 0.2967 while keeping")
print("round 1's standard-test gains, early stopping fixed the real problem -")
print("this becomes your production encoder. If frozen_recall is still stuck")
print("in the 0.24-0.29 range, overfitting wasn't the whole story and this")
print("branch is closed regardless of standard-test macro F1 looking good.")
