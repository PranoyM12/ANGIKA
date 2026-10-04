# src/evaluate.py
"""Evaluate the ANGIKA neural-network classifier on a held-out test split."""

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

MODEL_PATH = "models/mp_mlp_model.pkl"


def evaluate_model(model_path, features, labels):
    """Evaluate using the same fixed stratified 80/20 split used in training."""
    bundle = joblib.load(model_path)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]

    _, X_test, _, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy


if __name__ == "__main__":
    features = np.load("data/mp_features.npy")
    labels = np.load("data/mp_labels.npy")
    evaluate_model(MODEL_PATH, features, labels)
