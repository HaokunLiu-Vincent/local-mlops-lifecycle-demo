
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

# Generate a synthetic binary classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=5,
    n_redundant=0,
    random_state=42,
)

# Convert the dataset into a DataFrame
df = pd.DataFrame(
    X,
    columns=[f"f{i}" for i in range(8)],
)

df["label"] = y
df["row_id"] = range(len(df))

# Save raw data
output_path = Path("data/raw/dataset.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print(f"Dataset generated: {df.shape}")
print(df.head())
