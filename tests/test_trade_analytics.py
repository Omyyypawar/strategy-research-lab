"""Tests for the trade analytics module."""

import numpy as np
import pandas as pd
import pytest
import os
import tempfile

from src.trade_analytics import (
    extract_trades_from_log,
    compute_trade_metrics,
    rolling_metrics,
    _compute_streaks,
)


@pytest.fixture
def mock_log_csv(tmp_path):
    """Create a mock executor log CSV with BUY/SELL/HOLD actions."""
    data = {
        "timestamp": pd.date_range("2025-01-01", periods=20, freq="1min"),
        "open": [100] * 20,
        "high": [101] * 20,
        "low": [99] * 20,
        "close": [100 + i * 0.5 for i in range(20)],
        "volume": [1000] * 20,
        "action": [
            "HOLD", "BUY", "HOLD", "HOLD", "SELL",  # trade 1: entry@100.5, exit@102.0
            "HOLD", "BUY", "HOLD", "SELL", "HOLD",   # trade 2: entry@103.0, exit@104.0
            "BUY", "HOLD", "HOLD", "SELL", "HOLD",    # trade 3: entry@105.0, exit@106.5
            "BUY", "HOLD", "SELL", "HOLD", "HOLD",    # trade 4: entry@107.5, exit@108.5
        ],
        "qty": [0, 0.1, 0, 0, 0.1, 0, 0.1, 0, 0.1, 0,
                0.1, 0, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0],
        "signal_prob": [0.5] * 20,
    }
    df = pd.DataFrame(data)
    path = tmp_path / "test_log.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_trades():
    """Pre-built trades DataFrame for metric tests."""
    return pd.DataFrame({
        "entry_time": pd.date_range("2025-01-01", periods=10, freq="5min"),
        "exit_time": pd.date_range("2025-01-01 00:05", periods=10, freq="5min"),
        "entry_price": [100, 101, 99, 102, 98, 103, 97, 104, 96, 105],
        "exit_price": [101, 100, 100, 101, 99, 102, 98, 103, 97, 106],
        "qty": [0.1] * 10,
        "pnl": [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1],
        "pnl_pct": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0],
        "mae": [-0.5, -1.0, -0.3, -1.2, -0.4, -1.1, -0.2, -0.9, -0.5, -0.1],
        "mfe": [1.5, 0.5, 1.2, 0.3, 1.1, 0.4, 1.3, 0.2, 0.8, 1.5],
        "holding_seconds": [300] * 10,
        "entry_signal_prob": [0.55, 0.48, 0.62, 0.45, 0.58, 0.43, 0.65, 0.41, 0.52, 0.70],
    })


class TestExtractTrades:
    def test_extracts_correct_count(self, mock_log_csv):
        trades = extract_trades_from_log(mock_log_csv)
        assert len(trades) == 4  # 4 BUY/SELL pairs

    def test_has_required_columns(self, mock_log_csv):
        trades = extract_trades_from_log(mock_log_csv)
        required = ["entry_time", "exit_time", "entry_price", "exit_price", "pnl"]
        for col in required:
            assert col in trades.columns

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_trades_from_log("nonexistent.csv")


class TestComputeTradeMetrics:
    def test_returns_dict(self, sample_trades):
        metrics = compute_trade_metrics(sample_trades)
        assert isinstance(metrics, dict)

    def test_correct_trade_count(self, sample_trades):
        metrics = compute_trade_metrics(sample_trades)
        assert metrics["total_trades"] == 10

    def test_win_rate_bounded(self, sample_trades):
        metrics = compute_trade_metrics(sample_trades)
        assert 0 <= metrics["win_rate"] <= 1

    def test_profit_factor_positive(self, sample_trades):
        metrics = compute_trade_metrics(sample_trades)
        assert metrics["profit_factor"] > 0

    def test_empty_trades(self):
        metrics = compute_trade_metrics(pd.DataFrame())
        assert "error" in metrics


class TestStreaks:
    def test_all_wins(self):
        result = _compute_streaks(np.array([1, 1, 1, 1]))
        assert result["max_wins"] == 4
        assert result["max_losses"] == 0

    def test_alternating(self):
        result = _compute_streaks(np.array([1, -1, 1, -1]))
        assert result["max_wins"] == 1
        assert result["max_losses"] == 1

    def test_mixed(self):
        result = _compute_streaks(np.array([1, 1, -1, -1, -1, 1]))
        assert result["max_wins"] == 2
        assert result["max_losses"] == 3


class TestRollingMetrics:
    def test_returns_dataframe(self, sample_trades):
        result = rolling_metrics(sample_trades, window=5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_too_few_trades(self):
        small = pd.DataFrame({"pnl": [1], "pnl_pct": [1]})
        result = rolling_metrics(small, window=5)
        assert result.empty
