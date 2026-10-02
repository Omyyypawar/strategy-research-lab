"""Tests for the SQLite feature store."""

import os
import pytest
import numpy as np
import pandas as pd

from src.feature_store import FeatureStore


@pytest.fixture
def store(tmp_path):
    """Create a temporary feature store."""
    db_path = str(tmp_path / "test_features.db")
    return FeatureStore(db_path=db_path)


class TestFeatureStore:
    def test_log_and_count(self, store):
        store.log_features(
            timestamp="2025-01-01T00:00:00",
            symbol="BTCUSDT",
            features={"ret_1m": 0.001, "vol_short": 0.02},
            prediction=1,
            prediction_prob=0.65,
        )
        assert store.count_records("BTCUSDT") == 1

    def test_log_multiple(self, store):
        for i in range(10):
            store.log_features(
                timestamp=f"2025-01-01T00:{i:02d}:00",
                symbol="BTCUSDT",
                features={"ret_1m": 0.001 * i, "vol_short": 0.02},
            )
        assert store.count_records("BTCUSDT") == 10

    def test_get_recent_features(self, store):
        for i in range(5):
            store.log_features(
                timestamp=f"2025-01-01T00:{i:02d}:00",
                symbol="BTCUSDT",
                features={"f1": float(i), "f2": float(i * 2)},
                prediction=i % 2,
            )
        df = store.get_recent_features("BTCUSDT", limit=3)
        assert len(df) == 3  # only last 3
        assert "f1" in df.columns
        assert "f2" in df.columns

    def test_update_actual_label(self, store):
        store.log_features(
            timestamp="2025-01-01",
            symbol="BTCUSDT",
            features={"f1": 1.0},
            prediction=1,
        )
        store.update_actual_label(feature_id=1, actual_label=0)

        df = store.get_recent_features("BTCUSDT")
        assert df.iloc[0]["actual_label"] == 0

    def test_get_training_data_empty(self, store):
        # no labeled data -> returns None
        result = store.get_training_data("BTCUSDT", min_samples=1)
        assert result is None

    def test_compute_feature_stats(self, store):
        for i in range(20):
            store.log_features(
                timestamp=f"2025-01-01T00:{i:02d}:00",
                symbol="BTCUSDT",
                features={"f1": float(i), "f2": np.random.normal(0, 1)},
            )
        stats = store.compute_feature_stats("BTCUSDT")
        assert "f1" in stats
        assert "mean" in stats["f1"]
        assert "std" in stats["f1"]

    def test_drift_snapshot_and_detect(self, store):
        # log two snapshots
        store.log_drift_snapshot(
            {"f1": {"mean": 0.0, "std": 1.0, "min": -3, "max": 3, "n_samples": 100}},
            timestamp="2025-01-01",
        )
        store.log_drift_snapshot(
            {"f1": {"mean": 5.0, "std": 1.0, "min": 2, "max": 8, "n_samples": 100}},
            timestamp="2025-01-02",
        )
        drifted = store.detect_drift("BTCUSDT", threshold=2.0)
        assert "f1" in drifted  # mean shifted by 5 std devs

    def test_different_symbols_isolated(self, store):
        store.log_features("2025-01-01", "BTCUSDT", {"f1": 1.0})
        store.log_features("2025-01-01", "ETHUSDT", {"f1": 2.0})
        assert store.count_records("BTCUSDT") == 1
        assert store.count_records("ETHUSDT") == 1
