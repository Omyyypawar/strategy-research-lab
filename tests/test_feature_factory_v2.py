"""Tests for the production feature factory (V2)."""

import numpy as np
import pandas as pd
import pytest

from src.feature_factory_v2 import (
    build_features,
    safe_entropy,
    rolling_entropy,
    volume_imbalance,
    vpin_like,
    volatility_regime,
    zscore,
)


EXPECTED_FEATURE_COLS = [
    "ret_1m", "ret_5m", "vol_short", "vol_long", "vol_regime",
    "entropy_50", "norm_range_20", "vol_imbalance_20", "vpin_50",
    "zscore_ret5", "mom1", "mom2", "mom2_smooth", "avg_vol_20",
]


class TestBuildFeatures:
    """Tests for the main build_features() pipeline."""

    def test_returns_dataframe(self, sample_ohlcv):
        result = build_features(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_all_feature_columns_present(self, sample_ohlcv):
        result = build_features(sample_ohlcv)
        for col in EXPECTED_FEATURE_COLS:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_preserves_ohlcv_columns(self, sample_ohlcv):
        result = build_features(sample_ohlcv)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_no_nans_after_build(self, sample_ohlcv):
        result = build_features(sample_ohlcv)
        assert not result.isna().any().any(), "build_features should fill all NaNs"

    def test_row_count_preserved(self, sample_ohlcv):
        result = build_features(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)

    def test_does_not_mutate_input(self, sample_ohlcv):
        original = sample_ohlcv.copy()
        build_features(sample_ohlcv)
        pd.testing.assert_frame_equal(sample_ohlcv, original)

    def test_handles_string_numeric_columns(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df["close"] = df["close"].astype(str)
        result = build_features(df)
        assert result["close"].dtype in [np.float64, np.float32]

    def test_small_input_does_not_crash(self, sample_ohlcv_small):
        result = build_features(sample_ohlcv_small)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_small)

    def test_vol_regime_is_numeric(self, sample_ohlcv):
        result = build_features(sample_ohlcv)
        assert pd.api.types.is_numeric_dtype(result["vol_regime"])


class TestHelperFunctions:
    """Tests for individual feature helper functions."""

    def test_safe_entropy_uniform(self):
        arr = np.random.uniform(0, 1, 100)
        e = safe_entropy(arr, bins=10)
        assert e > 0, "Entropy of uniform data should be positive"

    def test_safe_entropy_constant(self):
        arr = np.ones(100)
        e = safe_entropy(arr, bins=10)
        assert e >= 0

    def test_safe_entropy_with_nans(self):
        arr = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        e = safe_entropy(arr, bins=5)
        assert np.isfinite(e)

    def test_rolling_entropy_length(self, sample_ohlcv):
        result = rolling_entropy(sample_ohlcv["close"], window=20, bins=10)
        assert len(result) == len(sample_ohlcv)

    def test_volume_imbalance_range(self, sample_ohlcv):
        result = volume_imbalance(sample_ohlcv, window=20)
        assert result.min() >= -1.01  # allow small float imprecision
        assert result.max() <= 1.01

    def test_vpin_like_range(self, sample_ohlcv):
        result = vpin_like(sample_ohlcv, bucket_size=50)
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.min() >= 0.0
            assert valid.max() <= 1.0

    def test_volatility_regime_positive(self, sample_ohlcv):
        result = volatility_regime(sample_ohlcv)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_zscore_centered(self, sample_ohlcv):
        result = zscore(sample_ohlcv["close"], window=50)
        valid = result.dropna()
        assert abs(valid.mean()) < 2.0  # roughly centered
