"""
experiment_runner_v2.py
Quick training + validation runner for Feature Factory V2 dataset.
Trains RandomForest and XGBoost (if available), prints metrics, saves models.
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/features_v2.parquet"
RF_MODEL_OUT = "models/phase3_rf_v2.pkl"
XGB_MODEL_OUT = "models/phase3_xgb_v2.json"

def load_data():
    df = pd.read_parquet(DATA_PATH)
    # drop timestamp, future columns if present
    df = df.drop(columns=["timestamp","future_close_5m","future_return_5m"], errors="ignore")
    df = df.dropna()
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

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

def try_xgb(X_train, y_train, X_val, y_val):
    try:
        import xgboost as xgb
    except Exception:
        print("XGBoost not installed — skipping.")
        return None

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {"objective":"binary:logistic","eval_metric":"auc","eta":0.05,"max_depth":6}
    evals = [(dval, "val")]
    bst = xgb.train(params, dtrain, num_boost_round=500, evals=evals, early_stopping_rounds=25, verbose_eval=False)
    return bst

def main():
    os.makedirs("models", exist_ok=True)
    print("Loading dataset...")
    X, y = load_data()
    # time-series safe split
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

    print("Training RandomForest...")
    rf = train_rf(X_train, y_train)
    preds = rf.predict(X_val)
    print("RF Accuracy:", accuracy_score(y_val, preds))
    print("RF F1:", f1_score(y_val, preds))
    print(classification_report(y_val, preds))
    joblib.dump(rf, RF_MODEL_OUT)
    print("Saved RF →", RF_MODEL_OUT)

    print("Attempting XGBoost...")
    xgb_model = try_xgb(X_train, y_train, X_val, y_val)
    if xgb_model is not None:
        xgb_model.save_model(XGB_MODEL_OUT)
        print("Saved XGB →", XGB_MODEL_OUT)
        # predict
        import xgboost as xgb
        predp = (xgb_model.predict(xgb.DMatrix(X_val)) > 0.5).astype(int)
        print("XGB Accuracy:", accuracy_score(y_val, predp))
        print("XGB F1:", f1_score(y_val, predp))

if __name__ == "__main__":
    main()


