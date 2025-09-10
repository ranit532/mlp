import pandas as pd
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42,
)

# Create a pandas DataFrame
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["target"] = y

# Save the dataset to a CSV file
df.to_csv("data/classification_data.csv", index=False)

print("Dataset created and saved to data/classification_data.csv")