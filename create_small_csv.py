import pandas as pd

# Load first 100 rows from mixed_categories_reviews.csv
df = pd.read_csv('test_data/mixed_categories_reviews.csv').head(100)
df.to_csv('test_data/small_test.csv', index=False)
