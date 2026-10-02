"""
experiment_runner_v2.py
Quick training + validation runner for Feature Factory V2 dataset.
Trains RandomForest and XGBoost (if available), prints metrics, saves models.
"""

import os
import joblib
import json
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
DATA_PATH = "data/features_v2.parquet"
RF_MODEL_OUT = "models/phase3_rf_v2.pkl"
XGB_MODEL_OUT = "models/phase3_xgb_v2.json"
META_OUT = "models/phase3_model_meta.json"
FEATURE_MANIFEST_OUT = "models/features_columns_phase3_v2.json"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
def load_data():
    df = pd.read_parquet(DATA_PATH)

    df = df.drop(columns=["timestamp", "future_close_5m", "future_return_5m"],
                 errors="ignore")

    df = df.dropna()

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


# ---------------------------------------------------------
# TRAIN RF
# ---------------------------------------------------------
def train_rf(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    return rf


# ---------------------------------------------------------
# TRY XGBOOST
# ---------------------------------------------------------
def try_xgb(X_train, y_train, X_val, y_val):
    try:
        import xgboost as xgb
    except Exception:
        print("XGBoost not installed — skipping.")
        return None

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.05,
        "max_depth": 6
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=25,
        verbose_eval=False
    )
    return bst


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    os.makedirs("models", exist_ok=True)

    print("Loading dataset...")
    X, y = load_data()

    # time-series split
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

    # Train RF
    print("Training RandomForest...")
    rf = train_rf(X_train, y_train)

    preds = rf.predict(X_val)
    print("RF Accuracy:", accuracy_score(y_val, preds))
    print("RF F1:", f1_score(y_val, preds))
    print(classification_report(y_val, preds))

    joblib.dump(rf, RF_MODEL_OUT)
    print("Saved RF →", RF_MODEL_OUT)

    # ---------------------------------------------------------
    # SAVE METADATA
    # ---------------------------------------------------------
    meta = {
        "model_name": "RandomForest_v2",
        "trained_on": DATA_PATH,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_columns": list(X.columns),
        "hyperparams": {
            "n_estimators": 500,
            "max_depth": 12,
            "min_samples_leaf": 3,
            "class_weight": "balanced_subsample"
        },
        "train_accuracy": float(accuracy_score(y_train, rf.predict(X_train))),
        "val_accuracy": float(accuracy_score(y_val, preds)),
        "val_f1": float(f1_score(y_val, preds))
    }

    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=4)

    print("Saved metadata →", META_OUT)

    # ---------------------------------------------------------
    # SAVE FEATURE MANIFEST
    # ---------------------------------------------------------
    with open(FEATURE_MANIFEST_OUT, "w") as f:
        json.dump(list(X_train.columns), f, indent=4)

    print("Saved feature manifest →", FEATURE_MANIFEST_OUT)

    # ---------------------------------------------------------
    # TRY XGB
    # ---------------------------------------------------------
    print("Attempting XGBoost...")
    xgb_model = try_xgb(X_train, y_train, X_val, y_val)

    if xgb_model is not None:
        xgb_model.save_model(XGB_MODEL_OUT)
        print("Saved XGB →", XGB_MODEL_OUT)


if __name__ == "__main__":
    main()



