# src/train.py
"""Train the ANGIKA neural-network pose classifier.

The original project used a linear SVM.  This version replaces it with a
scikit-learn MLP (feed-forward neural network) and StandardScaler.  The model
is saved as one Pipeline so prediction uses exactly the same preprocessing
that was used during training.
"""

import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


MODEL_PATH = "models/mp_mlp_model.pkl"


def build_model():
    """Create the preprocessing + neural-network pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=32,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])


def train_mlp(features, labels):
    """Train the neural-network classifier and report held-out accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    model = build_model()
    model.fit(X_train, y_train_encoded)

    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Validation/Test Accuracy: {accuracy:.2f}")

    return {"model": model, "label_encoder": label_encoder}, X_test, y_test, y_pred


if __name__ == "__main__":
    features = np.load("data/mp_features.npy")
    labels = np.load("data/mp_labels.npy")

    model_bundle, _, _, _ = train_mlp(features, labels)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
