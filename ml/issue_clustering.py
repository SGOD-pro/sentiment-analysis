"""
Issue detection via clustering - no labeled training data needed.

How this works, plainly:
1. Take embeddings of NEGATIVE reviews only (issues live in complaints)
2. KMeans groups reviews that sit close together in embedding space
3. You read a few examples per cluster and manually name it
   (e.g. cluster 3 = mostly shipping complaints -> name it "delivery")
4. New reviews get assigned to whichever cluster centroid is closest

This does NOT require pairs, triplets, or any similarity labels - it's
unsupervised, using embeddings you already have and already trust.

Two things you have to do that a script can't do for you:
- Pick K (number of clusters) - the elbow method below gives you a
  starting point, but there's real judgment involved, not a formula
- Read cluster examples and assign real names - this is the "labeling"
  step, just much cheaper than labeling every individual review
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

ROOT = Path("/content/drive/MyDrive/Dataset/embeddings_output")

df = pd.read_parquet(ROOT / "bge_clean_metadata.parquet")
X = np.load(ROOT / "bge_clean_embeddings.npy")

# --- only cluster negative reviews - that's where actionable issues live ---
negative_mask = df["label"] == "negative"
neg_df = df[negative_mask].reset_index(drop=True)
neg_X = X[negative_mask.values]

print(f"Negative reviews to cluster: {len(neg_df)}")

# ---------------------------------------------------------------------------
# STEP 1: elbow method to help pick K - not a hard answer, a starting point
# ---------------------------------------------------------------------------
print("\nRunning elbow method (this takes a few minutes, testing K=5 to K=30)...")
inertias = []
K_range = range(5, 31, 5)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(neg_X)
    inertias.append(km.inertia_)
    print(f"K={k}: inertia={km.inertia_:.1f}")

plt.figure(figsize=(8, 5))
plt.plot(list(K_range), inertias, marker="o")
plt.xlabel("K (number of clusters)")
plt.ylabel("Inertia (lower = tighter clusters)")
plt.title("Elbow method - look for where the curve stops dropping steeply")
plt.savefig(ROOT / "elbow_plot.png")
print(f"\nElbow plot saved to elbow_plot.png - look for the 'bend' in the curve.")
print("That's roughly where adding more clusters stops meaningfully helping.")
print("For a business issue taxonomy, K=10-20 is a reasonable real-world range -")
print("more than that gets too granular to act on, fewer loses useful distinction.")

# ---------------------------------------------------------------------------
# STEP 2: fit final KMeans with your chosen K
# ---------------------------------------------------------------------------
K = 15  # ADJUST based on the elbow plot - this is a starting guess, not final
print(f"\nFitting final KMeans with K={K}...")
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(neg_X)
neg_df["cluster"] = cluster_labels

# ---------------------------------------------------------------------------
# STEP 3: for each cluster, extract top TF-IDF terms (helps you guess the
# topic FAST without reading every single review) + sample reviews to
# actually confirm/read
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("CLUSTER SUMMARIES - use these to name each cluster")
print("="*60)

cluster_summary_rows = []

for cluster_id in range(K):
    cluster_texts = neg_df[neg_df["cluster"] == cluster_id]["clean_text"].fillna("").tolist()
    cluster_size = len(cluster_texts)

    if cluster_size < 5:
        print(f"\nCluster {cluster_id}: only {cluster_size} reviews, skipping TF-IDF (too small)")
        continue

    # top terms via TF-IDF, comparing this cluster's vocabulary against the
    # whole negative-review corpus - highlights what's DISTINCTIVE about
    # this cluster, not just common words like "bad" or "product"
    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
    all_neg_texts = neg_df["clean_text"].fillna("").tolist()
    tfidf_matrix = vectorizer.fit_transform(all_neg_texts)

    cluster_mask = (neg_df["cluster"] == cluster_id).values
    cluster_tfidf_mean = np.asarray(tfidf_matrix[cluster_mask].mean(axis=0)).flatten()
    top_indices = cluster_tfidf_mean.argsort()[-10:][::-1]
    feature_names = vectorizer.get_feature_names_out()
    top_terms = [feature_names[i] for i in top_indices]

    print(f"\n--- Cluster {cluster_id} (n={cluster_size}) ---")
    print(f"Top terms: {', '.join(top_terms)}")
    print("Sample reviews:")
    for text in cluster_texts[:3]:
        print(f"  - {text[:150]}")

    cluster_summary_rows.append({
        "cluster_id": cluster_id,
        "size": cluster_size,
        "top_terms": ", ".join(top_terms),
        "suggested_name": "",  # YOU fill this in after reading
    })

summary_df = pd.DataFrame(cluster_summary_rows)
summary_df.to_csv(ROOT / "cluster_naming_worksheet.csv", index=False)
print(f"\n\nSaved worksheet to cluster_naming_worksheet.csv - fill in")
print("'suggested_name' for each cluster based on the top terms and samples above.")

# save the fitted kmeans model + cluster assignments for reuse
import joblib
joblib.dump(kmeans, ROOT / "issue_kmeans_model.joblib")
neg_df[["id", "category", "clean_text", "cluster"]].to_csv(
    ROOT / "negative_reviews_clustered.csv", index=False
)
print("Saved issue_kmeans_model.joblib (for assigning new reviews to clusters)")
print("Saved negative_reviews_clustered.csv (full cluster assignments)")
