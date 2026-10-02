"""Tests for the incremental (streaming) feature engine."""

import numpy as np
import pytest

from src.incremental_features import RollingState, IncrementalFeatureEngine


class TestRollingState:
    def test_append_and_length(self):
        rs = RollingState(maxlen=100)
        for i in range(50):
            rs.append(i, i + 1, i - 1, i + 0.5, 1000)
        assert len(rs.closes) == 50

    def test_maxlen_enforced(self):
        rs = RollingState(maxlen=10)
        for i in range(20):
            rs.append(i, i, i, i, i)
        assert len(rs.closes) == 10

    def test_as_series_keys(self):
        rs = RollingState()
        rs.append(1, 2, 0.5, 1.5, 100)
        series = rs.as_series()
        assert set(series.keys()) == {"open", "high", "low", "close", "volume"}


class TestIncrementalFeatureEngine:
    def test_returns_dict(self):
        engine = IncrementalFeatureEngine()
        for i in range(100):
            result = engine.append_tick(100 + i * 0.1, 101 + i * 0.1, 99 + i * 0.1, 100.5 + i * 0.1, 500)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        engine = IncrementalFeatureEngine()
        for i in range(100):
            result = engine.append_tick(100, 101, 99, 100.5, 500)
        expected_keys = [
            "ret_1m", "ret_5m", "ret_15m",
            "ema_fast", "ema_slow", "ema_trend", "ema_trend_norm",
            "rolling_std_5", "atr", "zscore", "volume_spike",
            "price_range_ratio", "vol_regime",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_ema_updates_incrementally(self):
        engine = IncrementalFeatureEngine()
        for i in range(50):
            engine.append_tick(100, 101, 99, 100.0, 500)
        assert engine.ema_fast is not None
        assert engine.ema_slow is not None
        assert engine.ema_fast == pytest.approx(100.0, abs=0.5)

    def test_single_tick_does_not_crash(self):
        engine = IncrementalFeatureEngine()
        result = engine.append_tick(100, 101, 99, 100.5, 500)
        assert isinstance(result, dict)
