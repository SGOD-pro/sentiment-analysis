"""
Evaluation script to compare the existing global clustering vs the new per-category clustering.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib

ROOT = Path("/content/drive/MyDrive/Dataset/embeddings_output")
EXPORT_DIR = Path("./lambda_deploy_artifacts")

def main():
    print("Loading datasets...")
    df = pd.read_parquet(ROOT / "bge_clean_metadata.parquet")
    X = np.load(ROOT / "bge_clean_embeddings.npy")
    
    neg_mask = df["label"] == "negative"
    neg_df = df[neg_mask].reset_index(drop=True)
    neg_X = X[neg_mask.values]
    
    print("Loading clustering models...")
    
    # Load fallback KMeans model
    kmeans_fallback = joblib.load(ROOT / "issue_kmeans_model.joblib")
    
    # Load configuration to get distance thresholds
    config_path = EXPORT_DIR / "config.json"
    if not config_path.exists():
        print("config.json not found in lambda_deploy_artifacts. Run export_mlp_and_clusters.py first.")
        return
        
    with open(config_path) as f:
        config = json.load(f)
        
    issue_cfg = config["issue_detection"]
    fallback_threshold = issue_cfg.get("fallback_clusters", issue_cfg).get("distance_threshold", 0.70)
    per_cat_configs = issue_cfg.get("per_category_clusters", {})
    
    print("\n--- Fallback Clustering (Global) ---")
    
    fallback_labels = kmeans_fallback.predict(neg_X)
    fallback_distances = np.linalg.norm(neg_X - kmeans_fallback.cluster_centers_[fallback_labels], axis=1)
    
    num_other_fallback = (fallback_distances > fallback_threshold).sum()
    pct_other_fallback = num_other_fallback / len(neg_df) * 100
    
    print(f"Total Negative Reviews: {len(neg_df)}")
    print(f"Avg Distance to Centroid: {fallback_distances.mean():.4f}")
    print(f"Assigned to 'other' (>{fallback_threshold}): {num_other_fallback} ({pct_other_fallback:.1f}%)")
    
    print("\n--- Per-Category Clustering (New) ---")
    
    if not per_cat_configs:
        print("No per-category clusters found in config.json. Did you run the per_category_clustering script and export?")
        return
        
    new_distances = []
    new_labels = []
    cluster_sources = []
    
    categories_using_per_cat = list(per_cat_configs.keys())
    print(f"Categories using per-category clusters: {len(categories_using_per_cat)}")
    print(f"Categories using fallback: {len(neg_df['category'].unique()) - len(categories_using_per_cat)}")
    
    for i, row in neg_df.iterrows():
        cat = row["category"]
        emb = neg_X[i]
        
        if cat in per_cat_configs:
            model_path = ROOT / f"issue_kmeans_model_{cat.replace(' ', '_')}.joblib"
            if not hasattr(main, "loaded_models"):
                main.loaded_models = {}
            if cat not in main.loaded_models:
                main.loaded_models[cat] = joblib.load(model_path)
                
            km = main.loaded_models[cat]
            dist = np.linalg.norm(emb - km.cluster_centers_, axis=1)
            nearest_idx = int(dist.argmin())
            nearest_dist = dist[nearest_idx]
            
            threshold = per_cat_configs[cat].get("distance_threshold", fallback_threshold)
            if nearest_dist > threshold:
                label = "other"
            else:
                label = per_cat_configs[cat]["cluster_names"].get(str(nearest_idx), "other")
                
            new_distances.append(nearest_dist)
            new_labels.append(label)
            cluster_sources.append("per_category")
        else:
            # Fallback
            dist = np.linalg.norm(emb - kmeans_fallback.cluster_centers_, axis=1)
            nearest_idx = int(dist.argmin())
            nearest_dist = dist[nearest_idx]
            
            if nearest_dist > fallback_threshold:
                label = "other"
            else:
                label = issue_cfg.get("fallback_clusters", issue_cfg)["cluster_names"].get(str(nearest_idx), "other")
                
            new_distances.append(nearest_dist)
            new_labels.append(label)
            cluster_sources.append("cross_category_fallback")
            
    new_distances = np.array(new_distances)
    num_other_new = sum(1 for l in new_labels if l == "other")
    pct_other_new = num_other_new / len(neg_df) * 100
    
    print(f"Avg Distance to Centroid: {new_distances.mean():.4f}")
    print(f"Assigned to 'other': {num_other_new} ({pct_other_new:.1f}%)")
    
    print("\n--- Summary of Improvements ---")
    dist_diff = fallback_distances.mean() - new_distances.mean()
    print(f"Distance reduction (tighter clusters): {dist_diff:.4f}")
    
if __name__ == "__main__":
    main()
