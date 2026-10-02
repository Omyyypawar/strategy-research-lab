"""Tests for the ensemble model pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.ensemble_model import StackingEnsemble, make_random_forest


@pytest.fixture
def classification_data():
    """Generate synthetic binary classification data."""
    np.random.seed(42)
    n = 300
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
    })
    y = pd.Series(((X["f1"] + X["f2"] + np.random.randn(n) * 0.5) > 0).astype(int))
    return X, y


class TestStackingEnsemble:
    def test_fit_and_predict(self, classification_data):
        X, y = classification_data
        X_train, y_train = X.iloc[:200], y.iloc[:200]
        X_test, y_test = X.iloc[200:], y.iloc[200:]

        ensemble = StackingEnsemble(n_cv_folds=2)
        ensemble.fit(X_train, y_train)

        preds = ensemble.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, classification_data):
        X, y = classification_data
        ensemble = StackingEnsemble(n_cv_folds=2)
        ensemble.fit(X.iloc[:200], y.iloc[:200])

        proba = ensemble.predict_proba(X.iloc[200:])
        assert proba.shape == (100, 2)
        # probabilities should sum to ~1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0, decimal=5)

    def test_feature_names_stored(self, classification_data):
        X, y = classification_data
        ensemble = StackingEnsemble(n_cv_folds=2)
        ensemble.fit(X.iloc[:200], y.iloc[:200])
        assert ensemble.feature_names_in_ is not None
        assert len(ensemble.feature_names_in_) == 3

    def test_threshold_optimization(self, classification_data):
        X, y = classification_data
        ensemble = StackingEnsemble(n_cv_folds=2)
        ensemble.fit(
            X.iloc[:200], y.iloc[:200],
            X_val=X.iloc[200:250], y_val=y.iloc[200:250],
        )
        assert 0.3 <= ensemble.optimal_threshold <= 0.7

    def test_classes_attribute(self, classification_data):
        X, y = classification_data
        ensemble = StackingEnsemble(n_cv_folds=2)
        ensemble.fit(X.iloc[:200], y.iloc[:200])
        assert hasattr(ensemble, "classes_")
        assert 0 in ensemble.classes_
        assert 1 in ensemble.classes_

    def test_accuracy_above_random(self, classification_data):
        X, y = classification_data
        X_train, y_train = X.iloc[:200], y.iloc[:200]
        X_test, y_test = X.iloc[200:], y.iloc[200:]

        ensemble = StackingEnsemble(n_cv_folds=2)
        ensemble.fit(X_train, y_train)
        preds = ensemble.predict(X_test)

        accuracy = (preds == y_test.values).mean()
        assert accuracy > 0.45  # should beat random on this easy synthetic data
