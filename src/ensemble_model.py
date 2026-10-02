"""
Ensemble Model Pipeline — Strategy Research Lab
================================================
Multi-model stacking ensemble that combines:
- LightGBM (gradient boosting — fast, handles noisy features)
- RandomForest (bagging — robust baseline)
- Logistic Regression (meta-learner for stacking)

Key advantages over single RF:
- LightGBM captures non-linear interactions with lower overfitting risk
- Stacking combines uncorrelated model errors for better generalization
- Calibrated probabilities via meta-learner for confidence-aware sizing
- Built-in threshold optimization on validation set

Author: Om | Strategy Research Lab
"""

import os
import json
import time
import joblib
import logging
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, roc_auc_score, brier_score_loss,
)
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger("ensemble_model")

# ------------------------------------------------------------------
# Try importing LightGBM
# ------------------------------------------------------------------
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. Install with: pip install lightgbm")


# ------------------------------------------------------------------
# Base model factories
# ------------------------------------------------------------------
def make_random_forest():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )


def make_lightgbm():
    if not LGBM_AVAILABLE:
        raise ImportError("LightGBM not installed")
    return lgb.LGBMClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )


# ------------------------------------------------------------------
# Stacking Ensemble
# ------------------------------------------------------------------
class StackingEnsemble:
    """
    Two-layer stacking ensemble:
    Layer 1: RF + LightGBM (or RF-only fallback)
    Layer 2: Logistic Regression on out-of-fold probabilities
    """

    def __init__(self, n_cv_folds: int = 3):
        self.n_cv_folds = n_cv_folds
        self.base_models = []
        self.meta_model = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        self.feature_names_in_ = None
        self.classes_ = np.array([0, 1])
        self.optimal_threshold = 0.5
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        """
        Fit the stacking ensemble.
        Step 1: Generate out-of-fold predictions from base models
        Step 2: Train meta-learner on stacked OOF probabilities
        Step 3: Refit base models on full training data
        Step 4: Optimize threshold on validation set (if provided)
        """
        self.feature_names_in_ = np.array(X.columns.tolist())
        X_arr = X.values
        y_arr = y.values

        # define base models
        base_specs = [("rf", make_random_forest)]
        if LGBM_AVAILABLE:
            base_specs.append(("lgbm", make_lightgbm))

        n = len(X_arr)
        tscv = TimeSeriesSplit(n_splits=self.n_cv_folds)

        # Step 1: OOF predictions
        oof_preds = np.zeros((n, len(base_specs)))

        for model_idx, (name, factory) in enumerate(base_specs):
            logger.info("Generating OOF predictions for %s...", name)
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
                model = factory()
                model.fit(X_arr[train_idx], y_arr[train_idx])
                probs = model.predict_proba(X_arr[val_idx])
                oof_preds[val_idx, model_idx] = probs[:, 1]

        # Step 2: Train meta-learner on OOF probabilities
        # only use rows that have valid OOF predictions (not the first fold's train rows)
        valid_mask = oof_preds.sum(axis=1) != 0
        meta_X = oof_preds[valid_mask]
        meta_y = y_arr[valid_mask]

        logger.info("Training meta-learner on %d stacked samples...", len(meta_X))
        self.meta_model.fit(meta_X, meta_y)

        # Step 3: Refit all base models on full training data
        self.base_models = []
        for name, factory in base_specs:
            logger.info("Refitting %s on full training data...", name)
            model = factory()
            model.fit(X_arr, y_arr)
            self.base_models.append((name, model))

        # Step 4: Optimize threshold
        if X_val is not None and y_val is not None:
            self.optimal_threshold = self._optimize_threshold(X_val, y_val)
            logger.info("Optimal threshold: %.3f", self.optimal_threshold)

        self._fitted = True
        return self

    def _get_base_probabilities(self, X):
        """Get class-1 probabilities from each base model."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        probs = np.zeros((len(X_arr), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            probs[:, i] = model.predict_proba(X_arr)[:, 1]
        return probs

    def predict_proba(self, X):
        """Predict calibrated probabilities via meta-learner."""
        base_probs = self._get_base_probabilities(X)
        meta_probs = self.meta_model.predict_proba(base_probs)
        return meta_probs

    def predict(self, X):
        """Predict using optimal threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.optimal_threshold).astype(int)

    def _optimize_threshold(self, X_val, y_val, thresholds=None):
        """Find threshold that maximizes F1 on validation set."""
        if thresholds is None:
            thresholds = np.arange(0.40, 0.65, 0.01)

        proba = self.predict_proba(X_val)[:, 1]
        y_true = y_val.values if isinstance(y_val, pd.Series) else y_val

        best_f1 = -1
        best_t = 0.5

        for t in thresholds:
            preds = (proba >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        return float(best_t)


# ------------------------------------------------------------------
# Training pipeline
# ------------------------------------------------------------------
def train_ensemble(
    data_path: str = "data/features_v2.parquet",
    model_out: str = "models/ensemble_v1.pkl",
    meta_out: str = "models/ensemble_meta.json",
    manifest_out: str = "models/features_columns_phase3_v2.json",
):
    """Full ensemble training pipeline."""
    logging.basicConfig(level=logging.INFO)

    print("Loading dataset...")
    df = pd.read_parquet(data_path)
    df = df.drop(columns=["timestamp", "future_close_5m", "future_return_5m"], errors="ignore")
    df = df.dropna()

    X = df.drop(columns=["target"])
    y = df["target"]

    # Time-series split: 70% train, 15% val, 15% test
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Train ensemble
    ensemble = StackingEnsemble(n_cv_folds=3)
    t0 = time.time()
    ensemble.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    train_time = time.time() - t0

    # Evaluate on test set
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    print("\n" + "=" * 50)
    print("ENSEMBLE TEST RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  ROC AUC   : {auc:.4f}")
    print(f"  Brier     : {brier:.4f}")
    print(f"  Threshold : {ensemble.optimal_threshold:.3f}")
    print(f"  Train time: {train_time:.1f}s")
    print(f"\n  Base models: {[name for name, _ in ensemble.base_models]}")
    print(classification_report(y_test, y_pred))
    print("=" * 50)

    # Also evaluate individual base models for comparison
    print("\nPer-model comparison on test set:")
    for name, model in ensemble.base_models:
        bp = model.predict(X_test.values)
        ba = accuracy_score(y_test, bp)
        bf = f1_score(y_test, bp, zero_division=0)
        print(f"  {name:>8}: Acc={ba:.4f}  F1={bf:.4f}")
    print(f"  {'ensemble':>8}: Acc={acc:.4f}  F1={f1:.4f}")

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(ensemble, model_out)
    print(f"\nSaved ensemble model -> {model_out}")

    # Save metadata
    meta = {
        "model_name": "StackingEnsemble_v1",
        "base_models": [name for name, _ in ensemble.base_models],
        "meta_learner": "LogisticRegression",
        "optimal_threshold": ensemble.optimal_threshold,
        "n_cv_folds": ensemble.n_cv_folds,
        "trained_on": data_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_features": X.shape[1],
        "feature_columns": list(X.columns),
        "train_time_seconds": round(train_time, 1),
        "test_metrics": {
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "roc_auc": round(auc, 4),
            "brier_score": round(brier, 4),
        },
    }

    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Saved metadata -> {meta_out}")

    # Update feature manifest
    with open(manifest_out, "w") as f:
        json.dump(list(X.columns), f, indent=4)
    print(f"Saved feature manifest -> {manifest_out}")

    return ensemble


if __name__ == "__main__":
    train_ensemble()
