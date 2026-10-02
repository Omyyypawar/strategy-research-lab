"""Tests for the alternative data module."""

import numpy as np
import pytest

from src.alt_data import AltDataState, ALT_DATA_FEATURE_NAMES


class TestAltDataState:
    def test_empty_state_returns_defaults(self):
        state = AltDataState()
        feats = state.compute_features()
        assert isinstance(feats, dict)
        assert feats["funding_rate"] == 0.0
        assert feats["open_interest"] == 0.0
        assert feats["liq_count"] == 0
        assert feats["long_short_ratio"] == 1.0

    def test_funding_rate_features(self):
        state = AltDataState()
        for i in range(20):
            state.append_funding_rate(0.0001 * (i + 1))
        feats = state.compute_features()
        assert feats["funding_rate"] == pytest.approx(0.002)
        assert feats["funding_rate_ma5"] > 0
        assert "funding_rate_zscore" in feats

    def test_open_interest_features(self):
        state = AltDataState()
        for i in range(20):
            state.append_open_interest(1000 + i * 10)
        feats = state.compute_features()
        assert feats["open_interest"] == 1190.0
        assert feats["oi_change_5"] > 0  # increasing OI
        assert "oi_zscore" in feats

    def test_liquidation_tracking(self):
        state = AltDataState()
        # simulate 5 liquidations in current interval
        for _ in range(5):
            state.record_liquidation(1.0)
        state.flush_liquidations()
        # simulate 2 in next interval
        state.record_liquidation(1.0)
        state.record_liquidation(1.0)
        state.flush_liquidations()

        feats = state.compute_features()
        assert feats["liq_count"] == 2
        assert feats["liq_count_ma10"] > 0

    def test_long_short_ratio(self):
        state = AltDataState()
        for i in range(15):
            state.append_long_short_ratio(1.0 + i * 0.05)
        feats = state.compute_features()
        assert feats["long_short_ratio"] == pytest.approx(1.7)
        assert "ls_ratio_zscore" in feats

    def test_all_feature_names_present(self):
        state = AltDataState()
        # populate all buffers
        for i in range(20):
            state.append_funding_rate(0.0001)
            state.append_open_interest(1000)
            state.append_long_short_ratio(1.0)
            state.record_liquidation(1.0)
            state.flush_liquidations()

        feats = state.compute_features()
        for name in ALT_DATA_FEATURE_NAMES:
            assert name in feats, f"Missing alt data feature: {name}"

    def test_maxlen_enforced(self):
        state = AltDataState(maxlen=10)
        for i in range(50):
            state.append_funding_rate(i)
        assert len(state.funding_rates) == 10
