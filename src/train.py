import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def train():
    # Load and preprocess data
    data = pd.read_csv("data/classification_data.csv")
    X = data.drop("target", axis=1).values
    y = data["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Hyperparameters
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 1
    learning_rate = 0.001
    epochs = 100

    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("hidden_size", hidden_size)

        model = MLP(input_size, hidden_size, output_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        train_losses = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        mlflow.log_metric("final_train_loss", train_losses[-1])

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred_tensor = model(X_test)
            y_pred = y_pred_tensor.round().numpy()
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            print(f"Accuracy: {accuracy:.4f}")

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("loss_curve.png")
        mlflow.log_artifact("loss_curve.png")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # Log the model
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    train()