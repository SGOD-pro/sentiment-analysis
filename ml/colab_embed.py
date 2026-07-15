"""
Colab T4 GPU embedding pipeline
--------------------------------
Run this in a Colab cell (or as a .py via !python if you upload it).
Assumes: Runtime > Change runtime type > T4 GPU already selected.

Do NOT use multiprocessing here. One GPU, one process, big batches.
The GPU is already doing the parallel work internally - your only job
is to feed it efficiently.
"""

import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- sanity check first, always ---
assert torch.cuda.is_available(), (
    "No GPU detected. Runtime > Change runtime type > T4 GPU, then reconnect. "
    "Do not proceed on CPU - it will take 10-20x longer and this script "
    "isn't tuned for that path."
)
print(f"GPU: {torch.cuda.get_device_name(0)}")

ROOT = Path("/content")
INPUT_CSV = ROOT / "amazon_reviews.csv"   # upload this or mount Drive
TEXT_COLUMN = "text"

MODELS = {
    "bge": "BAAI/bge-small-en-v1.5",
    "mini": "sentence-transformers/all-MiniLM-L6-v2",
}

BATCH_SIZE = 512  # T4 has 16GB VRAM, these are small models - push it.
                   # If you OOM, drop to 256, don't go lower before trying fp16.


def embed_model(texts: list[str], model_name: str, out_file: Path) -> np.ndarray:
    """Encode all texts with one model, on GPU, fp16. sentence-transformers
    .encode() already sorts by length internally to minimize padding waste -
    that's another reason NOT to hand-chunk this yourself. Let the library
    do it."""
    model = SentenceTransformer(model_name, device="cuda")
    model.half()  # fp16 - Tensor Cores, ~2x speed, fine for embeddings

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device="cuda",
    )
    elapsed = time.time() - t0
    print(f"{model_name}: {len(texts)} texts in {elapsed:.1f}s "
          f"({len(texts)/elapsed:.0f} texts/sec)")

    np.save(out_file, embeddings)

    # free VRAM before loading the next model
    del model
    torch.cuda.empty_cache()

    return embeddings


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)

    if "id" not in df.columns:
        df.insert(0, "id", np.arange(len(df)))

    df.to_parquet(ROOT / "amazon_reviews.parquet", index=False)

    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
    print(f"Rows: {len(texts)}")

    print("\nEmbedding BGE...")
    embed_model(texts, MODELS["bge"], ROOT / "bge_embeddings.npy")

    print("\nEmbedding MiniLM...")
    embed_model(texts, MODELS["mini"], ROOT / "mini_embeddings.npy")

    print("\nDone. Files saved to /content/. Copy to Drive now if you want")
    print("them to survive the session ending:")
    print("  from google.colab import drive")
    print("  drive.mount('/content/drive')")
    print("  !cp /content/*.npy /content/*.parquet /content/drive/MyDrive/")
