"""
Online Learning Engine — Strategy Research Lab
===============================================
Per-candle model updates using River (incremental ML library).
This is the key differentiator: Freqtrade/3Commas do NOT do online updates.

Architecture:
- Wraps River classifiers behind a sklearn-compatible interface
- Maintains a rolling accuracy tracker for confidence
- Can run standalone or as a base model in StackingEnsemble

Supported models:
- HoeffdingTreeClassifier (default) — incremental decision tree
- AMFClassifier — Aggregated Mondrian Forest (online random forest)
- LogisticRegression — online logistic regression with SGD

Author: Om | Strategy Research Lab
"""

import numpy as np
import pandas as pd
import logging
import json
import pickle
import os
from collections import deque
from typing import Optional

logger = logging.getLogger("online_learner")

# ------------------------------------------------------------------
# Try importing River
# ------------------------------------------------------------------
try:
    from river import tree, ensemble, linear_model, metrics, preprocessing
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logger.warning("River not installed. Install with: pip install river")


# ------------------------------------------------------------------
# Online Model Wrapper
# ------------------------------------------------------------------
class OnlineLearner:
    """
    Sklearn-compatible wrapper around River online classifiers.
    Supports predict(), predict_proba(), and incremental learn_one().
    """

    def __init__(
        self,
        model_type: str = "hoeffding_tree",
        feature_names: Optional[list] = None,
        accuracy_window: int = 200,
    ):
        """
        Parameters
        ----------
        model_type : one of 'hoeffding_tree', 'amf', 'logistic'
        feature_names : list of feature column names (required for dict conversion)
        accuracy_window : rolling window size for accuracy tracking
        """
        if not RIVER_AVAILABLE:
            raise ImportError(
                "River is required for online learning. Install with: pip install river"
            )

        self.model_type = model_type
        self.feature_names = feature_names
        self.feature_names_in_ = np.array(feature_names) if feature_names else None
        self.classes_ = np.array([0, 1])

        # build River model
        self._model = self._build_model(model_type)

        # rolling accuracy tracker
        self._accuracy = metrics.Rolling(metrics.Accuracy(), window_size=accuracy_window)
        self._n_samples_seen = 0
        self._accuracy_window = accuracy_window

        # prediction history for diagnostics
        self._recent_preds = deque(maxlen=accuracy_window)
        self._recent_truths = deque(maxlen=accuracy_window)

    def _build_model(self, model_type: str):
        """Construct the River model pipeline."""
        scaler = preprocessing.StandardScaler()

        if model_type == "hoeffding_tree":
            base = tree.HoeffdingTreeClassifier(
                grace_period=50,
                max_depth=10,
                split_confidence=1e-5,
            )
        elif model_type == "amf":
            base = ensemble.AMFClassifier(
                n_estimators=10,
                seed=42,
            )
        elif model_type == "logistic":
            base = linear_model.LogisticRegression(
                optimizer=linear_model.base.optim.SGD(lr=0.01),
                l2=0.001,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # pipeline: scale -> classify
        return scaler | base

    def _row_to_dict(self, x):
        """Convert array/Series/dict to River-compatible dict."""
        if isinstance(x, dict):
            return {str(k): float(v) for k, v in x.items()}
        elif isinstance(x, pd.Series):
            return {str(k): float(v) for k, v in x.items()}
        elif isinstance(x, np.ndarray):
            if self.feature_names is not None:
                return {str(self.feature_names[i]): float(x[i]) for i in range(len(x))}
            else:
                return {str(i): float(x[i]) for i in range(len(x))}
        elif isinstance(x, pd.DataFrame):
            row = x.iloc[0]
            return {str(k): float(v) for k, v in row.items()}
        else:
            raise ValueError(f"Cannot convert {type(x)} to dict")

    def learn_one(self, x, y: int):
        """
        Update the model with a single observation.
        Call this after you know the true label (delayed feedback).

        Parameters
        ----------
        x : dict, Series, ndarray, or single-row DataFrame of features
        y : true binary label (0 or 1)
        """
        x_dict = self._row_to_dict(x)
        y_int = int(y)

        # update accuracy tracker with previous prediction
        if self._n_samples_seen > 0:
            # we predict first, then learn (prequential evaluation)
            y_pred = self._model.predict_one(x_dict)
            if y_pred is not None:
                self._accuracy.update(y_int, y_pred)

        # learn
        self._model.learn_one(x_dict, y_int)
        self._n_samples_seen += 1

    def predict(self, X) -> np.ndarray:
        """Predict class labels for X (DataFrame or array)."""
        if isinstance(X, pd.DataFrame):
            preds = []
            for _, row in X.iterrows():
                x_dict = self._row_to_dict(row)
                p = self._model.predict_one(x_dict)
                preds.append(int(p) if p is not None else 0)
            return np.array(preds)
        else:
            x_dict = self._row_to_dict(X)
            p = self._model.predict_one(x_dict)
            return np.array([int(p) if p is not None else 0])

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for X."""
        if isinstance(X, pd.DataFrame):
            probs = []
            for _, row in X.iterrows():
                x_dict = self._row_to_dict(row)
                p = self._model.predict_proba_one(x_dict)
                if p:
                    prob_1 = p.get(1, 0.5)
                    probs.append([1 - prob_1, prob_1])
                else:
                    probs.append([0.5, 0.5])
            return np.array(probs)
        else:
            x_dict = self._row_to_dict(X)
            p = self._model.predict_proba_one(x_dict)
            if p:
                prob_1 = p.get(1, 0.5)
                return np.array([[1 - prob_1, prob_1]])
            return np.array([[0.5, 0.5]])

    def predict_one(self, x) -> int:
        """Predict a single observation."""
        x_dict = self._row_to_dict(x)
        p = self._model.predict_one(x_dict)
        return int(p) if p is not None else 0

    def predict_proba_one(self, x) -> dict:
        """Predict probabilities for a single observation."""
        x_dict = self._row_to_dict(x)
        return self._model.predict_proba_one(x_dict) or {0: 0.5, 1: 0.5}

    @property
    def rolling_accuracy(self) -> float:
        """Current rolling accuracy (over last N samples)."""
        return self._accuracy.get() if self._n_samples_seen > 0 else 0.5

    @property
    def n_samples_seen(self) -> int:
        return self._n_samples_seen

    def get_stats(self) -> dict:
        """Return diagnostic stats."""
        return {
            "model_type": self.model_type,
            "n_samples_seen": self._n_samples_seen,
            "rolling_accuracy": round(self.rolling_accuracy, 4),
            "accuracy_window": self._accuracy_window,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Batch fit (for compatibility with StackingEnsemble).
        Iterates through data in order, calling learn_one.
        """
        self.feature_names = list(X.columns)
        self.feature_names_in_ = np.array(self.feature_names)

        for i in range(len(X)):
            self.learn_one(X.iloc[i], int(y.iloc[i]))

        return self

    def save(self, path: str):
        """Save model state to pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved online learner to %s", path)

    @staticmethod
    def load(path: str) -> "OnlineLearner":
        """Load model state from pickle."""
        with open(path, "rb") as f:
            return pickle.load(f)
