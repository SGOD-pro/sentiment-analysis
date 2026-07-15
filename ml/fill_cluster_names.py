import pandas as pd
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Dataset/embeddings_output")

df = pd.read_csv(ROOT / "cluster_naming_worksheet.csv")

# Based on the cluster summaries you already read - edit these names
# if any don't match what you saw in the actual sample reviews.
# Clusters 4, 6, 11 are the weak/noisy ones flagged earlier -
# both 4 and 11 are book/movie content complaints so one becomes
# content_quality, the other gets mapped to other to avoid duplicates.
# Cluster 6 was a genuine grab-bag, also other.
cluster_names = {
    0:  "sizing_and_fit",
    1:  "audio_and_music_quality",
    2:  "food_taste_and_pet",
    3:  "software_and_app_issues",
    4:  "content_quality",
    5:  "color_and_appearance",
    6:  "other",
    7:  "product_malfunction",
    8:  "general_dissatisfaction",
    9:  "value_and_price",
    10: "durability_and_build",
    11: "other",
    12: "scent_and_smell",
    13: "order_and_fulfillment",
    14: "breakage_and_damage",
}

df["suggested_name"] = df["cluster_id"].map(cluster_names)
df.to_csv(ROOT / "cluster_naming_worksheet.csv", index=False)

print("Saved. Verify before re-running export:")
print(df[["cluster_id", "suggested_name"]].to_string(index=False))
