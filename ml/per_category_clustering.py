"""
Per-category issue clustering for Phase 13.1.
Runs KMeans clustering on negative reviews separately for categories with >= 500 negative reviews.
Outputs `per_category_naming_worksheet.csv` for manual labeling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

ROOT = Path("/content/drive/MyDrive/Dataset/embeddings_output")
VOLUME_GATE = 500
K_CLUSTERS = 10

def main():
    print("Loading data...")
    df = pd.read_parquet(ROOT / "bge_clean_metadata.parquet")
    X = np.load(ROOT / "bge_clean_embeddings.npy")
    
    # Filter for negative reviews
    neg_mask = df["label"] == "negative"
    neg_df = df[neg_mask].reset_index(drop=True)
    neg_X = X[neg_mask.values]
    
    # Identify eligible categories
    category_counts = neg_df["category"].value_counts()
    eligible_categories = category_counts[category_counts >= VOLUME_GATE].index.tolist()
    
    print(f"Found {len(eligible_categories)} categories with >= {VOLUME_GATE} negative reviews:")
    for cat in eligible_categories:
        print(f" - {cat}: {category_counts[cat]} reviews")
        
    cluster_summary_rows = []
    
    for category in eligible_categories:
        print(f"\nProcessing category: {category}")
        cat_mask = (neg_df["category"] == category).values
        cat_X = neg_X[cat_mask]
        cat_df = neg_df[cat_mask].reset_index(drop=True)
        
        # Fit KMeans
        print(f"Fitting KMeans with K={K_CLUSTERS}...")
        kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cat_X)
        cat_df["cluster"] = cluster_labels
        
        # Save model
        model_path = ROOT / f"issue_kmeans_model_{category.replace(' ', '_')}.joblib"
        joblib.dump(kmeans, model_path)
        print(f"Saved model to {model_path}")
        
        # Extract TF-IDF terms
        for cluster_id in range(K_CLUSTERS):
            cluster_texts = cat_df[cat_df["cluster"] == cluster_id]["clean_text"].fillna("").tolist()
            cluster_size = len(cluster_texts)
            
            if cluster_size < 5:
                continue
                
            vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
            cat_texts = cat_df["clean_text"].fillna("").tolist()
            tfidf_matrix = vectorizer.fit_transform(cat_texts)
            
            c_mask = (cat_df["cluster"] == cluster_id).values
            cluster_tfidf_mean = np.asarray(tfidf_matrix[c_mask].mean(axis=0)).flatten()
            top_indices = cluster_tfidf_mean.argsort()[-10:][::-1]
            feature_names = vectorizer.get_feature_names_out()
            top_terms = [feature_names[i] for i in top_indices]
            
            cluster_summary_rows.append({
                "category": category,
                "cluster_id": cluster_id,
                "size": cluster_size,
                "top_terms": ", ".join(top_terms),
                "suggested_name": "",  # To be filled by user
            })
            
    summary_df = pd.DataFrame(cluster_summary_rows)
    worksheet_path = ROOT / "per_category_naming_worksheet.csv"
    summary_df.to_csv(worksheet_path, index=False)
    print(f"\nDone! Saved worksheet to {worksheet_path}")
    print("Please fill in the 'suggested_name' column for each cluster before exporting.")

if __name__ == "__main__":
    main()
