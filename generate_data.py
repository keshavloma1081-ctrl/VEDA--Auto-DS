"""
Generate a 10M row synthetic customer review dataset
for testing VEDA on large-scale text data.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

print("Generating 10M row synthetic text dataset...")
print("This will take 2-3 minutes...")

random.seed(42)
np.random.seed(42)

# ── Text templates ────────────────────────────────────────────
positive_reviews = [
    "absolutely love this product amazing quality",
    "great service fast delivery highly recommend",
    "excellent product worth every penny",
    "fantastic experience will buy again",
    "outstanding quality very happy with purchase",
    "best product I have ever used",
    "superb quality exceeded my expectations",
    "wonderful experience customer service was great",
    "incredible value for money highly satisfied",
    "perfect product exactly as described",
    "very happy with this purchase great quality",
    "amazing product fast shipping love it",
    "top quality product great value recommend",
    "brilliant product works perfectly very happy",
    "excellent quality fast delivery recommend to everyone",
]

negative_reviews = [
    "terrible product broke after one day",
    "very disappointed with this purchase",
    "worst product ever complete waste of money",
    "poor quality do not recommend",
    "horrible experience customer service useless",
    "product stopped working after one week",
    "completely disappointed with this item",
    "bad quality not worth the price",
    "awful product nothing like the description",
    "terrible experience will never buy again",
    "very poor quality broke immediately",
    "disgusting product horrible customer service",
    "waste of money do not buy",
    "dreadful quality very unhappy with purchase",
    "shocking product completely useless",
]

neutral_reviews = [
    "product is okay nothing special average quality",
    "decent product does the job nothing more",
    "average experience not great not terrible",
    "product works as expected nothing special",
    "okay quality for the price",
    "standard product meets basic requirements",
    "acceptable quality average delivery time",
    "product is fine does what it says",
    "mediocre quality average value for money",
    "neither good nor bad just average",
]

categories = ["Electronics", "Clothing", "Food", "Books", "Sports",
              "Beauty", "Home", "Toys", "Automotive", "Garden"]

print("Building rows...")

BATCH_SIZE = 500_000
TOTAL_ROWS = 10_000_000
N_BATCHES = TOTAL_ROWS // BATCH_SIZE

os.makedirs("data", exist_ok=True)
output_path = "data/reviews_10m.csv"

# Write in batches to avoid OOM
first_batch = True
for batch_num in range(N_BATCHES):
    sentiments = np.random.choice(["positive", "negative", "neutral"],
                                   size=BATCH_SIZE,
                                   p=[0.5, 0.3, 0.2])

    reviews = []
    labels = []
    for s in sentiments:
        if s == "positive":
            base = random.choice(positive_reviews)
            label = 1
        elif s == "negative":
            base = random.choice(negative_reviews)
            label = 0
        else:
            base = random.choice(neutral_reviews)
            label = 0
        # Add slight variation
        words = base.split()
        if random.random() > 0.7:
            words = words + random.choice(positive_reviews).split()[:3]
        reviews.append(" ".join(words))
        labels.append(label)

    batch_df = pd.DataFrame({
        "review_id": range(batch_num * BATCH_SIZE, (batch_num + 1) * BATCH_SIZE),
        "review_text": reviews,
        "sentiment": labels,
        "category": np.random.choice(categories, size=BATCH_SIZE),
        "rating": np.random.randint(1, 6, size=BATCH_SIZE),
        "helpful_votes": np.random.randint(0, 100, size=BATCH_SIZE),
    })

    batch_df.to_csv(output_path,
                    mode="w" if first_batch else "a",
                    header=first_batch,
                    index=False)
    first_batch = False

    progress = (batch_num + 1) / N_BATCHES * 100
    print("Progress: " + str(round(progress)) + "% — batch " +
          str(batch_num + 1) + "/" + str(N_BATCHES) + " written")

# Verify
df_check = pd.read_csv(output_path, nrows=5)
file_size = os.path.getsize(output_path) / (1024 ** 3)
print("\nDataset generated successfully!")
print("Path      : " + output_path)
print("File size : " + str(round(file_size, 2)) + " GB")
print("Rows      : 10,000,000")
print("Columns   : " + str(len(df_check.columns)))
print("Columns   : " + str(list(df_check.columns)))