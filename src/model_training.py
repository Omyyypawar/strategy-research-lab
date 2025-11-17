"""
model_training.py
Strategy Research Lab — Phase 3

Trains a binary classifier using the SAME features used in live trading.
Outputs:
- models/phase3_rf_v2.pkl
"""
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)

DATA_PATH = "data/features_v2.parquet"
SAVE_RF = "models/phase3_rf_v2.pkl"

def load_dataset():
    print("Loading dataset...")
    df = pd.read_parquet(DATA_PATH)

    # Drop timestamp & future columns
    df = df.drop(columns=["timestamp", "future_close_5m", "future_return_5m"], errors="ignore")

    df = df.dropna()

    print("Dataset loaded:", df.shape)
    return df

def train_random_forest(X_train, y_train):
    print("\nTraining RandomForestClassifier...")

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model

def main():
    df = load_dataset()

    # Split features & target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Train/Validation split (time-series safe)
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    print("\nTraining dataset:", X_train.shape)
    print("Validation dataset:", X_val.shape)

    # Train RF
    rf = train_random_forest(X_train, y_train)

    # Validation
    preds = rf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)

    print("\n=== RANDOM FOREST RESULTS ===")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print(classification_report(y_val, preds))

    # Save RF
    joblib.dump(rf, SAVE_RF)
    print(f"\nSaved RandomForest model → {SAVE_RF}")


if __name__ == "__main__":
    main()

