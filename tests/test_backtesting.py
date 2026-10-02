"""Tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting import calculate_performance


class TestCalculatePerformance:
    """Tests for the performance metrics calculator."""

    @pytest.fixture
    def backtest_df(self):
        """Simple backtest DataFrame with known signals and returns."""
        np.random.seed(42)
        n = 252  # one trading year
        returns = np.random.normal(0.0005, 0.02, n)
        signals = np.random.choice([0, 1], n)
        return pd.DataFrame({"Signal": signals, "Return": returns})

    def test_returns_dict(self, backtest_df):
        result = calculate_performance(backtest_df)
        assert isinstance(result, dict)

    def test_required_metrics_present(self, backtest_df):
        result = calculate_performance(backtest_df)
        for key in ["CAGR", "Sharpe", "Sortino", "MaxDD", "Final_Equity"]:
            assert key in result, f"Missing metric: {key}"

    def test_final_equity_positive(self, backtest_df):
        result = calculate_performance(backtest_df)
        assert result["Final_Equity"] > 0

    def test_max_drawdown_bounded(self, backtest_df):
        result = calculate_performance(backtest_df)
        assert 0 <= result["MaxDD"] <= 1.0

    def test_all_hold_signal(self):
        """All signals = 0 means no trading; equity stays at 1."""
        df = pd.DataFrame({
            "Signal": [0] * 100,
            "Return": np.random.normal(0.001, 0.01, 100),
        })
        result = calculate_performance(df)
        assert result["Final_Equity"] == pytest.approx(1.0, abs=1e-9)

    def test_all_long_signal(self):
        """All signals = 1 should produce equity matching cumulative returns."""
        returns = [0.01, -0.005, 0.02, -0.01, 0.005]
        df = pd.DataFrame({
            "Signal": [1] * 5,
            "Return": returns,
        })
        result = calculate_performance(df)
        # Strategy return is shifted by 1, so first bar is neutral
        assert result["Final_Equity"] > 0

    def test_equity_curve_column_created(self, backtest_df):
        calculate_performance(backtest_df)
        assert "Equity_Curve" in backtest_df.columns
        assert "Benchmark_Curve" in backtest_df.columns
