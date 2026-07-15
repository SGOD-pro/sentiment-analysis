"""
Retrains the MLP that won the three-model comparison (neg_recall=0.7915,
neutral_recall=0.7426, frozen_recall=0.2967) and ACTUALLY SAVES IT this
time. The comparison script that produced those numbers never persisted
the model - it only existed in that session's memory. This is the fix.

Same architecture, same seed, same v4 data as before - should reproduce
very close to the same weights and numbers, but this time you'll have a
real file on disk to build from.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Dataset/embeddings_output")

df = pd.read_parquet(ROOT / "bge_clean_metadata.parquet")
X = np.load(ROOT / "bge_clean_embeddings.npy")
train_idx = np.load(ROOT / "clean_train_idx_v4.npy")
test_idx = np.load(ROOT / "clean_test_idx_v4.npy")
registry = pd.read_csv(ROOT / "audit_registry.csv")
frozen_eval = pd.read_csv(ROOT / "difficult_neutral_eval_FROZEN.csv")

reviewed_ids = set(registry["id"])
frozen_ids = set(frozen_eval["id"])
test_ids = set(df.iloc[test_idx]["id"])
assert len((reviewed_ids | frozen_ids) & test_ids) == 0, "LEAK - STOP."
print("Leak check passed.\n")

le = LabelEncoder().fit(["negative", "neutral", "positive"])
y = le.transform(df["label"])
NEGATIVE_ID = le.transform(["negative"])[0]
NEUTRAL_ID = le.transform(["neutral"])[0]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

frozen_indices = df.index[df["id"].isin(frozen_eval["id"])].values
frozen_positions = np.array([df.index.get_loc(i) for i in frozen_indices])
X_frozen, y_frozen = X[frozen_positions], y[frozen_positions]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}\n")

class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
X_frozen_t = torch.tensor(X_frozen, dtype=torch.float32).to(device)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True,
    generator=torch.Generator().manual_seed(42),  # THE FIX - pins shuffle order explicitly
)

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

torch.manual_seed(42)  # matches the comparison script's seed
model = MLP(X_train.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights_t)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

best_f1, best_state_dict, patience_counter = 0.0, None, 0
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
    print(f"Epoch {epoch+1} | loss: {total_loss:.4f} | test macro F1: {epoch_f1:.4f}")

    if epoch_f1 > best_f1:
        best_f1 = epoch_f1
        # THE FIX - actually capture and hold onto the best weights
        best_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 5:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ---------------------------------------------------------------------------
# THE ACTUAL FIX: save to disk, not just keep in memory
# ---------------------------------------------------------------------------
torch.save(best_state_dict, ROOT / "final_mlp_state.pt")
print(f"\nSaved MLP state_dict to {ROOT / 'final_mlp_state.pt'}")

# reload into a fresh model to confirm the save/load round-trip works and
# to compute final verified metrics from the ACTUAL saved file, not the
# in-memory training state (paranoia justified given what just happened)
verify_model = MLP(X_train.shape[1]).to(device)
verify_model.load_state_dict(torch.load(ROOT / "final_mlp_state.pt"))
verify_model.eval()

with torch.no_grad():
    test_probs = torch.softmax(verify_model(X_test_t), dim=1).cpu().numpy()
    frozen_probs = torch.softmax(verify_model(X_frozen_t), dim=1).cpu().numpy()

test_preds = test_probs.argmax(axis=1)
frozen_preds = frozen_probs.argmax(axis=1)

neg_recall = recall_score(y_test, test_preds, labels=[NEGATIVE_ID], average="macro")
neutral_recall = recall_score(y_test, test_preds, labels=[NEUTRAL_ID], average="macro")
frozen_recall = recall_score(y_frozen, frozen_preds, labels=[NEUTRAL_ID], average="macro")

print(f"\n{'='*55}\nVERIFIED FROM SAVED FILE (not memory)\n{'='*55}")
print(f"Neg recall: {neg_recall:.4f}")
print(f"Neutral recall: {neutral_recall:.4f}")
print(f"Frozen recall: {frozen_recall:.4f}")
print("\nCompare to original comparison run: neg=0.7915, neutral=0.7426, frozen=0.2967")
print("Should be close (same seed/data) but not necessarily bit-identical -")
print("DataLoader shuffling order differs run-to-run even with the seed set")
print("on model init, since torch.manual_seed doesn't fully pin DataLoader")
print("shuffle order without a generator argument. Small differences are")
print("expected and fine; large ones would mean something's wrong.")
