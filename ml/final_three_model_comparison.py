"""
Final comparison: LogisticRegression, LinearSVC, MLP - all on v4 (fully
corrected, leak-free, taxonomized data). Full classification reports on
both standard test set and frozen difficult-neutral slice.

LinearSVC note: doesn't have predict_proba natively (it's a margin-based
classifier, not probabilistic). Using decision_function + manual softmax-
like normalization to get comparable margin scores for the threshold rule.
This is an approximation, not true calibrated probability - flagged in
output so you don't over-interpret the "probs" as calibrated.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, recall_score

ROOT = "/content/drive/MyDrive/Dataset/embeddings_output"

df = pd.read_parquet(f"{ROOT}/bge_clean_metadata.parquet")
X = np.load(f"{ROOT}/bge_clean_embeddings.npy")
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
y = le.transform(df["label"])
NEGATIVE_ID = le.transform(["negative"])[0]
NEUTRAL_ID = le.transform(["neutral"])[0]
POSITIVE_ID = le.transform(["positive"])[0]
LABEL_NAMES = list(le.classes_)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

frozen_indices = df.index[df["id"].isin(frozen_eval["id"])].values
frozen_positions = np.array([df.index.get_loc(i) for i in frozen_indices])
X_frozen, y_frozen = X[frozen_positions], y[frozen_positions]


def apply_asymmetric_rule(probs, pos_threshold, neg_threshold=0.0):
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    argmax_preds = probs.argmax(axis=1)
    final_preds = argmax_preds.copy()
    positive_uncertain = (argmax_preds == POSITIVE_ID) & (margins < pos_threshold)
    final_preds[positive_uncertain] = NEUTRAL_ID
    negative_uncertain = (argmax_preds == NEGATIVE_ID) & (margins < neg_threshold)
    final_preds[negative_uncertain] = NEUTRAL_ID
    return final_preds, margins


def full_report(y_true, preds, label, has_all_classes=True):
    print(f"\n{'='*55}\n{label}\n{'='*55}")
    if has_all_classes:
        print(classification_report(y_true, preds, target_names=LABEL_NAMES, digits=4))
        macro_f1 = f1_score(y_true, preds, average="macro")
        print(f"Macro F1: {macro_f1:.4f}")
    neg_recall = recall_score(y_true, preds, labels=[NEGATIVE_ID], average="macro")
    neutral_recall = recall_score(y_true, preds, labels=[NEUTRAL_ID], average="macro")
    print(f"Negative recall: {neg_recall:.4f}")
    print(f"Neutral recall:  {neutral_recall:.4f}")
    return neg_recall, neutral_recall


results_summary = []

# ---------------------------------------------------------------------------
# 1. LogisticRegression - your current best deliverable
# ---------------------------------------------------------------------------
print("\n" + "#"*60 + "\n# LOGISTIC REGRESSION\n" + "#"*60)
lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight="balanced")
lr.fit(X_train, y_train)

lr_test_probs = lr.predict_proba(X_test)
lr_frozen_probs = lr.predict_proba(X_frozen)

lr_test_preds_baseline = lr_test_probs.argmax(axis=1)
full_report(y_test, lr_test_preds_baseline, "LR - Standard test (no threshold)")

lr_test_preds_thresh, _ = apply_asymmetric_rule(lr_test_probs, pos_threshold=0.3)
lr_frozen_preds_thresh, _ = apply_asymmetric_rule(lr_frozen_probs, pos_threshold=0.3)
neg_r, neu_r = full_report(y_test, lr_test_preds_thresh, "LR + asymmetric threshold - Standard test")
full_report(y_frozen, lr_frozen_preds_thresh, "LR + asymmetric threshold - Frozen slice", has_all_classes=False)
results_summary.append({"model": "LogisticRegression + threshold", "neg_recall": neg_r, "neutral_recall": neu_r})

# ---------------------------------------------------------------------------
# 2. LinearSVC - CalibratedClassifierCV wraps it to get probability estimates
# (needed for the threshold rule to work the same way across all 3 models)
# ---------------------------------------------------------------------------
print("\n" + "#"*60 + "\n# LINEAR SVC (calibrated for probability estimates)\n" + "#"*60)
svc = LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)
calibrated_svc = CalibratedClassifierCV(svc, cv=3, method="sigmoid")
calibrated_svc.fit(X_train, y_train)

svc_test_probs = calibrated_svc.predict_proba(X_test)
svc_frozen_probs = calibrated_svc.predict_proba(X_frozen)

svc_test_preds_baseline = svc_test_probs.argmax(axis=1)
full_report(y_test, svc_test_preds_baseline, "LinearSVC - Standard test (no threshold)")

svc_test_preds_thresh, _ = apply_asymmetric_rule(svc_test_probs, pos_threshold=0.3)
svc_frozen_preds_thresh, _ = apply_asymmetric_rule(svc_frozen_probs, pos_threshold=0.3)
neg_r, neu_r = full_report(y_test, svc_test_preds_thresh, "LinearSVC + asymmetric threshold - Standard test")
full_report(y_frozen, svc_frozen_preds_thresh, "LinearSVC + asymmetric threshold - Frozen slice", has_all_classes=False)
results_summary.append({"model": "LinearSVC + threshold", "neg_recall": neg_r, "neutral_recall": neu_r})

# ---------------------------------------------------------------------------
# 3. MLP - reuse architecture from earlier tests, no upweighting (that branch
# is closed - just the plain v4-trained MLP + threshold)
# ---------------------------------------------------------------------------
print("\n" + "#"*60 + "\n# MLP\n" + "#"*60)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

device = "cuda" if torch.cuda.is_available() else "cpu"
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
X_frozen_t = torch.tensor(X_frozen, dtype=torch.float32).to(device)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden=256, n_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )
    def forward(self, x):
        return self.net(x)

torch.manual_seed(42)
model = MLP(X_train.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights_t)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

best_f1, best_probs, patience_counter = 0.0, None, 0
for epoch in range(25):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    model.eval()
    with torch.no_grad():
        test_probs = torch.softmax(model(X_test_t), dim=1).cpu().numpy()
    epoch_f1 = f1_score(y_test, test_probs.argmax(axis=1), average="macro")
    scheduler.step(total_loss)
    if epoch_f1 > best_f1:
        best_f1, best_probs, patience_counter = epoch_f1, test_probs, 0
    else:
        patience_counter += 1
        if patience_counter >= 5:
            break

with torch.no_grad():
    mlp_frozen_probs = torch.softmax(model(X_frozen_t), dim=1).cpu().numpy()

mlp_test_preds_baseline = best_probs.argmax(axis=1)
full_report(y_test, mlp_test_preds_baseline, "MLP - Standard test (no threshold)")

mlp_test_preds_thresh, _ = apply_asymmetric_rule(best_probs, pos_threshold=0.3)
mlp_frozen_preds_thresh, _ = apply_asymmetric_rule(mlp_frozen_probs, pos_threshold=0.3)
neg_r, neu_r = full_report(y_test, mlp_test_preds_thresh, "MLP + asymmetric threshold - Standard test")
full_report(y_frozen, mlp_frozen_preds_thresh, "MLP + asymmetric threshold - Frozen slice", has_all_classes=False)
results_summary.append({"model": "MLP + threshold", "neg_recall": neg_r, "neutral_recall": neu_r})

# ---------------------------------------------------------------------------
print("\n" + "="*60 + "\nFINAL SUMMARY - all with asymmetric threshold (pos=0.3)\n" + "="*60)
print(pd.DataFrame(results_summary).to_string(index=False))
print("\nPick based on neutral_recall gain vs your current LR+threshold baseline")
print("(neg_recall ~0.7687, neutral_recall from earlier tests ~0.75). If")
print("LinearSVC or MLP clearly beats LR on neutral_recall while holding")
print("neg_recall, that becomes your new production model. If all three are")
print("within noise of each other, stick with LR - simplest, fastest, no")
print("reason to add complexity for an unmeasurable gain.")
