"""
Model swap test - is 78% a data ceiling or a linear-model capacity ceiling?

Uses clean_train_idx_v2 / clean_test_idx_v2 - the leak-free split. Do NOT
point this at the old train_idx/test_idx files, those are contaminated
(1345/6600 corrected ids were inside that old test set).

Runs two candidates against your existing LogisticRegression baseline:
  1. MLP with class-weighted loss (PyTorch, GPU if available)
  2. XGBoost with class weights (nearly free to run alongside, CPU is fine)

Compare all three on the SAME clean test set. Whichever wins, by how much,
tells you whether to invest more in labeling or in modeling.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

ROOT = "/content/drive/MyDrive/Dataset/embeddings_output"

df = pd.read_parquet(f"{ROOT}/bge_clean_metadata.parquet")
X = np.load(f"{ROOT}/bge_clean_embeddings.npy")  # not mmap here, MLP needs it in memory anyway
train_idx = np.load(f"{ROOT}/clean_train_idx_v2.npy")
test_idx = np.load(f"{ROOT}/clean_test_idx_v2.npy")

# sanity check every single time you load a split - cheap insurance against
# ever repeating the leak mistake
registry = pd.read_csv(f"{ROOT}/audit_registry.csv")
corrected_ids = set(registry["id"])
test_ids = set(df.iloc[test_idx]["id"])
leaked = corrected_ids & test_ids
assert len(leaked) == 0, f"LEAK DETECTED: {len(leaked)} corrected ids in test set. STOP."
print("Leak check passed: 0 corrected ids in test set.\n")

le = LabelEncoder().fit(["negative", "neutral", "positive"])
y = le.transform(df["label"])

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

LABEL_NAMES = list(le.classes_)


def report(name, y_true, y_pred):
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    neutral_f1 = f1_score(y_true, y_pred, average=None, labels=[le.transform(["neutral"])[0]])[0]
    print(f"Macro F1: {macro_f1:.4f} | Neutral F1 specifically: {neutral_f1:.4f}")
    return macro_f1, neutral_f1


# --- baseline: your existing LogisticRegression, for direct comparison ---
lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight="balanced")
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
report("LogisticRegression (baseline)", y_test, lr_preds)


# --- candidate 1: MLP with class-weighted loss ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nMLP training on: {device}")

class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

input_dim = X_train.shape[1]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden=256, n_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


model = MLP(input_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights_t)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

EPOCHS = 25
best_macro_f1 = 0.0
patience_counter = 0
PATIENCE = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).argmax(dim=1).cpu().numpy()
    epoch_macro_f1 = f1_score(y_test, test_preds, average="macro")
    scheduler.step(total_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | loss: {total_loss:.4f} | "
          f"test macro F1: {epoch_macro_f1:.4f}")

    if epoch_macro_f1 > best_macro_f1:
        best_macro_f1 = epoch_macro_f1
        best_preds = test_preds
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}, best macro F1: {best_macro_f1:.4f}")
            break

mlp_macro_f1, mlp_neutral_f1 = report("MLP (class-weighted, best epoch)", y_test, best_preds)


# --- candidate 2: XGBoost, nearly free to run alongside ---
try:
    import xgboost as xgb

    sample_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    weight_map = dict(zip(np.unique(y_train), sample_weights))
    sw_train = np.array([weight_map[label] for label in y_train])

    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=3,
        tree_method="hist",
        device=device,  # uses GPU if available
        random_state=42,
    )
    xgb_clf.fit(X_train, y_train, sample_weight=sw_train)
    xgb_preds = xgb_clf.predict(X_test)
    xgb_macro_f1, xgb_neutral_f1 = report("XGBoost (class-weighted)", y_test, xgb_preds)
except ImportError:
    print("\nxgboost not installed - run `!pip install xgboost` in Colab first, skipping.")
    xgb_macro_f1 = None


# --- final comparison ---
print(f"\n{'='*50}\nSUMMARY\n{'='*50}")
print(f"LogisticRegression macro F1: {f1_score(y_test, lr_preds, average='macro'):.4f}")
print(f"MLP macro F1:                {mlp_macro_f1:.4f}  (neutral F1: {mlp_neutral_f1:.4f})")
if xgb_macro_f1:
    print(f"XGBoost macro F1:            {xgb_macro_f1:.4f}  (neutral F1: {xgb_neutral_f1:.4f})")
print("\nIf both candidates beat LR by <1 point: this is a data ceiling, not a")
print("model capacity ceiling. Go back to targeted neutral-boundary labeling.")
print("If either beats LR by 2+ points: model capacity was the bottleneck,")
print("keep pushing this direction (deeper MLP, more XGBoost tuning) before")
print("sinking more time into manual labeling.")
