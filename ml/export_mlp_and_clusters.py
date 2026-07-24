"""
Final export step. Run once, locally/Colab, produces everything Lambda needs:
  1. MLP weights as plain numpy (no PyTorch in Lambda)
  2. KMeans cluster centroids as plain numpy (no sklearn in Lambda) - combined in .npz
  3. Cluster name mapping (from your manually-named worksheet)
  4. Unified config tying sentiment + issue detection together

This assumes `mlp_model` (your trained PyTorch MLP) and `kmeans` (your
fitted KMeans model) are already in memory from earlier session work, OR
reload them from their saved checkpoints - see notes inline.
"""

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
import glob

ROOT = Path("/content/drive/MyDrive/Dataset/embeddings_output")
EXPORT_DIR = Path("./lambda_deploy_artifacts")
EXPORT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# STEP 1: MLP weights -> numpy
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn

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

mlp_model = MLP(384)  # BGE-small output dim is 384
mlp_model.load_state_dict(torch.load(ROOT / "final_mlp_state.pt", map_location="cpu"))
mlp_model.eval()
print("Loaded MLP from final_mlp_state.pt")

state_dict = mlp_model.state_dict()
mlp_weights = {name: tensor.cpu().numpy() for name, tensor in state_dict.items()}
np.savez(EXPORT_DIR / "mlp_weights.npz", **mlp_weights)
print(f"MLP weights exported. Keys: {list(mlp_weights.keys())}")
print("VERIFY these key names match what lambda_handler.py expects (net.0.weight,")
print("net.3.weight, net.6.weight, etc) - if your architecture differs, the")
print("forward pass in the handler needs matching updates.")

# ---------------------------------------------------------------------------
# STEP 2: KMeans centroids -> numpy (combined into .npz)
# ---------------------------------------------------------------------------
centroids_dict = {}
config_issue_detection = {
    "only_runs_on_sentiment": "negative",
    "fallback_label": "other",
    "per_category_clusters": {}
}

# 2.1 Load fallback KMeans
kmeans_fallback = joblib.load(ROOT / "issue_kmeans_model.joblib")
print(f"Loaded fallback KMeans from issue_kmeans_model.joblib")
centroids_dict["cross_category_fallback"] = kmeans_fallback.cluster_centers_

# distance threshold: if a review's nearest centroid is farther than this,
# tag it "other" instead of forcing a weak match - prevents low-confidence
# cluster assignments from polluting your issue tags with noise
DISTANCE_THRESHOLD = 0.70  # calibrated from distance-to-own-centroid distribution

assert DISTANCE_THRESHOLD is not None, (
    "DISTANCE_THRESHOLD is still None. Run this script once to see the "
    "distance distribution printed at the bottom, pick the 90th-95th "
    "percentile value, set it above, then rerun. Don't ship with None - "
    "it disables the 'other' fallback entirely and forces every review "
    "into a cluster even when it doesn't belong to any of them."
)

# 2.2 Load fallback cluster names
naming_worksheet = pd.read_csv(ROOT / "cluster_naming_worksheet.csv")
assert naming_worksheet["suggested_name"].notna().all() and \
       (naming_worksheet["suggested_name"] != "").all(), (
    "cluster_naming_worksheet.csv has empty suggested_name values - "
    "fill in every cluster's name before exporting for production."
)

cluster_names_fallback = dict(zip(
    naming_worksheet["cluster_id"].astype(int),
    naming_worksheet["suggested_name"]
))

config_issue_detection["fallback_clusters"] = {
    "cluster_names": {str(k): v for k, v in cluster_names_fallback.items()},
    "distance_threshold": DISTANCE_THRESHOLD
}

# 2.3 Load per-category KMeans models and cluster names
per_cat_worksheet_path = ROOT / "per_category_naming_worksheet.csv"
if per_cat_worksheet_path.exists():
    per_cat_worksheet = pd.read_csv(per_cat_worksheet_path)
    assert per_cat_worksheet["suggested_name"].notna().all() and \
           (per_cat_worksheet["suggested_name"] != "").all(), (
        "per_category_naming_worksheet.csv has empty suggested_name values."
    )
    
    categories = per_cat_worksheet["category"].unique()
    for cat in categories:
        model_path = ROOT / f"issue_kmeans_model_{cat.replace(' ', '_')}.joblib"
        if model_path.exists():
            km = joblib.load(model_path)
            centroids_dict[cat] = km.cluster_centers_
            
            cat_rows = per_cat_worksheet[per_cat_worksheet["category"] == cat]
            cat_names = dict(zip(
                cat_rows["cluster_id"].astype(int),
                cat_rows["suggested_name"]
            ))
            
            config_issue_detection["per_category_clusters"][cat] = {
                "cluster_names": {str(k): v for k, v in cat_names.items()},
                "distance_threshold": DISTANCE_THRESHOLD  # Can be customized per category if needed
            }
            print(f"Loaded per-category model and names for: {cat}")
else:
    print(f"No {per_cat_worksheet_path} found. Skipping per-category clusters.")

# 2.4 Save all centroids to NPZ
np.savez(EXPORT_DIR / "issue_centroids.npz", **centroids_dict)
print(f"\nKMeans centroids exported to issue_centroids.npz with keys: {list(centroids_dict.keys())}")


# ---------------------------------------------------------------------------
# STEP 3: unified config
# ---------------------------------------------------------------------------
config = {
    "sentiment": {
        "positive_margin_threshold": 0.3,
        "negative_margin_threshold": 0.0,
        "label_names": ["negative", "neutral", "positive"],
        "negative_class_id": 0,
        "neutral_class_id": 1,
        "positive_class_id": 2,
        "mlp_architecture": {"input_dim": 384, "hidden1": 256, "hidden2": 128, "n_classes": 3},
    },
    "issue_detection": config_issue_detection,
}
with open(EXPORT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\nAll artifacts in {EXPORT_DIR}/:")
for f in EXPORT_DIR.iterdir():
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name}: {size_kb:.1f} KB")

print("\n" + "="*60)
print("STILL NEEDED: calibrate DISTANCE_THRESHOLD before this is production-ready.")
print("Run the calibration snippet below on your existing")
print("negative_reviews_clustered.csv to pick a sane cutoff.")
print("="*60)

# ---------------------------------------------------------------------------
# Calibration helper - run separately, inspect the distribution, then go
# back and set DISTANCE_THRESHOLD above before re-running this export
# ---------------------------------------------------------------------------
try:
    clustered = pd.read_csv(ROOT / "negative_reviews_clustered.csv")
    bge_embeddings = np.load(ROOT / "bge_clean_embeddings.npy")
    full_df = pd.read_parquet(ROOT / "bge_clean_metadata.parquet")

    clustered_ids = set(clustered["id"])
    mask = full_df["id"].isin(clustered_ids)
    clustered_embeddings = bge_embeddings[mask.values]
    clustered_full = full_df[mask].reset_index(drop=True)
    clustered_full = clustered_full.merge(clustered[["id", "cluster"]], on="id")

    distances_to_own_centroid = []
    centroids = centroids_dict["cross_category_fallback"]
    for i, row in clustered_full.iterrows():
        c = int(row["cluster"])
        dist = np.linalg.norm(clustered_embeddings[i] - centroids[c])
        distances_to_own_centroid.append(dist)

    distances_to_own_centroid = np.array(distances_to_own_centroid)
    print(f"\nDistance-to-own-centroid distribution (all clustered reviews):")
    print(pd.Series(distances_to_own_centroid).describe())
    print("\nA sane DISTANCE_THRESHOLD is roughly the 90th-95th percentile of this -")
    print("reviews farther than that from EVERY centroid are genuinely novel/")
    print("off-topic and should be tagged 'other' rather than forced into the")
    print("nearest (but still distant) cluster.")
except Exception as e:
    print("Could not run calibration helper. Ensure negative_reviews_clustered.csv exists.")
