"""
Retrain the MLP with human corrections merged into the v4 training set,
then compare against the frozen eval benchmark.

If the new model beats the current one on BOTH:
  - neg_recall  >= current neg_recall
  - frozen_recall >= current frozen_recall
then it replaces lambda/artifacts/mlp_weights.npz in-place.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

# ── Current model benchmark (from save_final_mlp.py verified run) ──────────
CURRENT_NEG_RECALL = 0.7915
CURRENT_FROZEN_RECALL = 0.2967

LABEL_NAMES = ["negative", "neutral", "positive"]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def embed_texts(texts: list[str], artifacts_dir: Path) -> np.ndarray:
    """Embed new correction texts using the local BGE ONNX model."""
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError:
        print("ERROR: onnxruntime / tokenizers not installed locally.")
        print("Run on Colab or use --skip-new-embeddings to only apply label flips.")
        sys.exit(1)

    bge_dir = artifacts_dir / "bge_onnx_quantized"
    tokenizer = Tokenizer.from_file(str(bge_dir / "tokenizer.json"))
    session = ort.InferenceSession(str(bge_dir / "model_quantized.onnx"))

    embeddings = []
    for text in texts:
        enc = tokenizer.encode(text, add_special_tokens=True)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        token_type = np.zeros_like(ids)
        out = session.run(None, {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": token_type,
        })
        # mean-pool over token dimension
        emb = out[0].mean(axis=1)
        embeddings.append(emb[0])
    return np.array(embeddings, dtype=np.float32)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, device: str) -> MLP:
    le = LabelEncoder().fit(LABEL_NAMES)
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    
    gen = torch.Generator()
    gen.manual_seed(42)
    
    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=256, shuffle=True,
        generator=gen,
    )

    torch.manual_seed(42)
    model = MLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_f1, best_state, patience_counter = 0.0, None, 0
    X_test_t = torch.tensor(X_train, dtype=torch.float32).to(device)  # use train for epoch F1 proxy

    for epoch in range(25):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        from sklearn.metrics import f1_score
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(X_test_t), dim=1).cpu().numpy()
        epoch_f1 = f1_score(y_train, probs.argmax(axis=1), average="macro")
        scheduler.step(total_loss)
        print(f"Epoch {epoch + 1:2d} | loss {total_loss:.4f} | train macro F1 {epoch_f1:.4f}")

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    return model


def evaluate(model: MLP, X: np.ndarray, y: np.ndarray, device: str, label: str) -> dict:
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)

    le = LabelEncoder().fit(LABEL_NAMES)
    neg_id = le.transform(["negative"])[0]
    neu_id = le.transform(["neutral"])[0]

    neg_recall = recall_score(y, preds, labels=[neg_id], average="macro")
    neu_recall = recall_score(y, preds, labels=[neu_id], average="macro")
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(y, preds, average="macro")

    print(f"\n{'=' * 55}\n{label}\n{'=' * 55}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Neg recall:     {neg_recall:.4f}  (current: {CURRENT_NEG_RECALL})")
    print(f"Neutral recall: {neu_recall:.4f}")
    return {"neg_recall": neg_recall, "frozen_recall": neu_recall, "macro_f1": macro_f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrections", required=True, help="CSV from export_corrections.py")
    parser.add_argument("--artifacts", default="lambda/artifacts", help="Lambda artifacts directory")
    parser.add_argument("--min-corrections", type=int, default=os.environ.get("MIN_CORRECTIONS", 10))
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    data_dir = Path(os.environ.get("DATA_DIR", "ml/training_data"))
    
    corrections = pd.read_csv(args.corrections)
    if len(corrections) < args.min_corrections:
        print(f"Skipping retrain: {len(corrections)} corrections found, minimum is {args.min_corrections}")
        sys.exit(0)
    
    print(f"Loaded {len(corrections)} corrections")

    # Load permanent data files
    df = pd.read_parquet(data_dir / "bge_clean_metadata.parquet")
    X = np.load(data_dir / "bge_clean_embeddings.npy")
    train_idx = np.load(data_dir / "clean_train_idx_v4.npy").tolist()
    test_idx = np.load(data_dir / "clean_test_idx_v4.npy").tolist()
    frozen_eval = pd.read_csv(data_dir / "difficult_neutral_eval_FROZEN.csv")

    corrected_metadata_path = data_dir / "bge_clean_metadata_corrected.parquet"
    if corrected_metadata_path.exists():
        print("WARNING: bge_clean_metadata_corrected.parquet exists! Using bge_clean_metadata.parquet as authoritative.")

    le = LabelEncoder().fit(LABEL_NAMES)

    new_texts, new_labels = [], []

    for _, row in corrections.iterrows():
        review_id = row["review_id"]
        # Look up review_id in df["id"]
        match_idx = df.index[df["id"] == review_id].tolist()
        
        if match_idx:
            # Update existing label
            df.loc[match_idx[0], "label"] = row["manual_label"]
        else:
            # New text
            new_texts.append(row["text"])
            new_labels.append(row["manual_label"])

    print(f"New correction texts to embed: {len(new_texts)}")
    
    if new_texts:
        new_X = embed_texts(new_texts, artifacts_dir)
        X = np.vstack([X, new_X])
        
        new_df = pd.DataFrame({
            "id": ["new_" + str(i) for i in range(len(new_texts))],
            "text": new_texts,
            "label": new_labels
        })
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Append new row index to train_idx ONLY
        start_new_idx = len(X) - len(new_texts)
        train_idx.extend(range(start_new_idx, len(X)))

    train_idx_arr = np.array(train_idx)
    test_idx_arr = np.array(test_idx)

    y_train = le.transform(df["label"].iloc[train_idx_arr].values)
    X_train = X[train_idx_arr]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on {device} with {len(X_train)} samples ({len(corrections) - len(new_texts)} existing modified, {len(new_texts)} new)")

    model = train_mlp(X_train, y_train, device)

    # Evaluate on standard test set
    y_test = le.transform(df["label"].iloc[test_idx_arr].values)
    test_metrics = evaluate(model, X[test_idx_arr], y_test, device, "Standard test set (v4 + corrections)")

    # Evaluate on frozen difficult-neutral slice
    frozen_indices = df.index[df["id"].isin(frozen_eval["id"])].values
    frozen_positions = np.array([df.index.get_loc(i) for i in frozen_indices])
    y_frozen = le.transform(df["label"].iloc[frozen_positions].values)
    frozen_metrics = evaluate(model, X[frozen_positions], y_frozen, device, "FROZEN difficult-neutral slice")

    beats_neg = test_metrics["neg_recall"] >= CURRENT_NEG_RECALL
    beats_frozen = frozen_metrics["frozen_recall"] >= CURRENT_FROZEN_RECALL

    print(f"\n{'=' * 55}")
    print(f"neg_recall improvement:    {'✓' if beats_neg else '✗'}  {test_metrics['neg_recall']:.4f} vs {CURRENT_NEG_RECALL}")
    print(f"frozen_recall improvement: {'✓' if beats_frozen else '✗'}  {frozen_metrics['frozen_recall']:.4f} vs {CURRENT_FROZEN_RECALL}")

    if beats_neg and beats_frozen:
        state_dict = model.cpu().state_dict()
        weights = {name: t.numpy() for name, t in state_dict.items()}
        out_path = artifacts_dir / "mlp_weights.npz"
        np.savez(str(out_path), **weights)
        print(f"\n✅  New model is better — artifacts replaced at {out_path}")
    else:
        print("\n⚠️  New model did not beat the current benchmark on both metrics.")
        sys.exit(1)


if __name__ == "__main__":
    main()
