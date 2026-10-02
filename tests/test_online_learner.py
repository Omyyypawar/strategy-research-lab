"""Tests for the online learning module."""

import numpy as np
import pandas as pd
import pytest

# Check if River is available
try:
    import river
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RIVER_AVAILABLE,
    reason="River not installed (pip install river)"
)


@pytest.fixture
def online_model():
    from src.online_learner import OnlineLearner
    return OnlineLearner(
        model_type="hoeffding_tree",
        feature_names=["f1", "f2", "f3"],
        accuracy_window=50,
    )


@pytest.fixture
def training_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
    })
    y = pd.Series(((X["f1"] + X["f2"]) > 0).astype(int))
    return X, y


class TestOnlineLearner:
    def test_learn_one_and_predict(self, online_model):
        # train on some samples
        for i in range(50):
            x = {"f1": np.random.randn(), "f2": np.random.randn(), "f3": np.random.randn()}
            y = 1 if x["f1"] + x["f2"] > 0 else 0
            online_model.learn_one(x, y)

        assert online_model.n_samples_seen == 50

        # predict
        x_test = {"f1": 1.0, "f2": 1.0, "f3": 0.0}
        pred = online_model.predict_one(x_test)
        assert pred in (0, 1)

    def test_predict_proba_shape(self, online_model):
        for i in range(30):
            online_model.learn_one({"f1": i, "f2": -i, "f3": 0}, i % 2)

        X_test = pd.DataFrame({"f1": [1.0], "f2": [-1.0], "f3": [0.0]})
        proba = online_model.predict_proba(X_test)
        assert proba.shape == (1, 2)
        assert abs(proba[0].sum() - 1.0) < 1e-6

    def test_batch_predict(self, online_model, training_data):
        X, y = training_data
        # train on first 100
        for i in range(100):
            online_model.learn_one(X.iloc[i], int(y.iloc[i]))

        # batch predict on next 100
        preds = online_model.predict(X.iloc[100:])
        assert len(preds) == 100
        assert set(np.unique(preds)).issubset({0, 1})

    def test_batch_fit(self, training_data):
        from src.online_learner import OnlineLearner
        X, y = training_data
        model = OnlineLearner(model_type="hoeffding_tree")
        model.fit(X, y)
        assert model.n_samples_seen == len(X)
        assert model.feature_names_in_ is not None

    def test_rolling_accuracy(self, online_model, training_data):
        X, y = training_data
        for i in range(100):
            online_model.learn_one(X.iloc[i], int(y.iloc[i]))
        acc = online_model.rolling_accuracy
        assert 0 <= acc <= 1

    def test_get_stats(self, online_model):
        for i in range(10):
            online_model.learn_one({"f1": i, "f2": 0, "f3": 0}, i % 2)
        stats = online_model.get_stats()
        assert stats["model_type"] == "hoeffding_tree"
        assert stats["n_samples_seen"] == 10

    def test_classes_attribute(self, online_model):
        assert hasattr(online_model, "classes_")
        assert 0 in online_model.classes_
        assert 1 in online_model.classes_

    def test_feature_names_in_attribute(self, online_model):
        assert online_model.feature_names_in_ is not None
        assert len(online_model.feature_names_in_) == 3
