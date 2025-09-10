
import mlflow
import mlflow.pytorch
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def predict_batch():
    # Load the model and scaler
    runs = mlflow.search_runs(experiment_ids=["0"])
    latest_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{latest_run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)

    # Load the training data to fit the scaler
    data = pd.read_csv("data/classification_data.csv")
    X = data.drop("target", axis=1).values
    scaler = StandardScaler()
    scaler.fit(X)

    # Load some data to predict on (e.g., the first 5 rows of the test set)
    X_sample = X[:5]
    y_sample = data["target"].values[:5]

    # Preprocess the data
    scaled_features = scaler.transform(X_sample)
    tensor_features = torch.tensor(scaled_features, dtype=torch.float32)

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(tensor_features)
        predictions = predictions.round().numpy().flatten()

    print("Sample predictions:")
    for i in range(len(predictions)):
        print(f"  Actual: {y_sample[i]}, Predicted: {predictions[i]}")

if __name__ == "__main__":
    predict_batch()
