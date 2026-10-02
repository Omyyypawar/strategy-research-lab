"""Tests for Phase 2 technical indicators."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import SMA, EMA, RSI, MACD, Bollinger_Bands


class TestSMA:
    def test_length_preserved(self):
        s = pd.Series(range(50), dtype=float)
        assert len(SMA(s, 10)) == 50

    def test_constant_series(self):
        s = pd.Series([5.0] * 20)
        result = SMA(s, 10)
        assert result.iloc[-1] == pytest.approx(5.0)


class TestEMA:
    def test_length_preserved(self):
        s = pd.Series(range(50), dtype=float)
        assert len(EMA(s, 10)) == 50

    def test_ema_converges_to_constant(self):
        s = pd.Series([10.0] * 100)
        result = EMA(s, 20)
        assert result.iloc[-1] == pytest.approx(10.0, abs=0.01)


class TestRSI:
    def test_rsi_range(self):
        np.random.seed(42)
        s = pd.Series(np.random.normal(0, 1, 200).cumsum() + 100)
        rsi = RSI(s, 14)
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_rsi_overbought_on_uptrend(self):
        s = pd.Series(np.linspace(100, 200, 100))
        rsi = RSI(s, 14)
        assert rsi.iloc[-1] >= 50  # consistent uptrend should be at or above 50


class TestMACD:
    def test_macd_returns_three_series(self):
        s = pd.Series(np.random.normal(0, 1, 100).cumsum() + 100)
        line, signal, hist = MACD(s)
        assert len(line) == 100
        assert len(signal) == 100
        assert len(hist) == 100

    def test_histogram_is_line_minus_signal(self):
        s = pd.Series(np.random.normal(0, 1, 100).cumsum() + 100)
        line, signal, hist = MACD(s)
        np.testing.assert_array_almost_equal(hist.values, (line - signal).values)


class TestBollingerBands:
    def test_upper_above_lower(self):
        s = pd.Series(np.random.normal(100, 5, 50))
        upper, mid, lower, _ = Bollinger_Bands(s, 20, 2)
        valid_idx = upper.dropna().index
        assert (upper[valid_idx] >= lower[valid_idx]).all()
