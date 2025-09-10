from fastapi import FastAPI
import mlflow
import mlflow.pytorch
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
from typing import List

# Define the input data model
class InputData(BaseModel):
    features: List[float]

# Initialize FastAPI
app = FastAPI()

# Load the model and scaler
def load_model_and_scaler():
    # Search for the latest run
    runs = mlflow.search_runs(experiment_ids=["0"])
    latest_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{latest_run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)

    # Load the training data to fit the scaler
    data = pd.read_csv("data/classification_data.csv")
    X = data.drop("target", axis=1).values
    scaler = StandardScaler()
    scaler.fit(X)

    return model, scaler

model, scaler = load_model_and_scaler()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    # Preprocess the input data
    features = np.array(data.features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    tensor_features = torch.tensor(scaled_features, dtype=torch.float32)

    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(tensor_features)
        prediction = prediction.round().item()

    return {"prediction": prediction}