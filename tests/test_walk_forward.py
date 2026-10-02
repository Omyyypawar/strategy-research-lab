"""Tests for the walk-forward backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from src.walk_forward import (
    walk_forward_backtest,
    WalkForwardResult,
    _sharpe,
    _max_drawdown,
    _compute_strategy_returns,
    _monte_carlo_test,
)


@pytest.fixture
def wf_dataset():
    """Generate synthetic feature + target dataset for WF testing."""
    np.random.seed(42)
    n = 500
    base_price = 100.0
    returns = np.random.normal(0, 0.01, n)
    close = base_price * np.cumprod(1 + returns)

    X = pd.DataFrame({
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
        "feat_3": np.random.randn(n),
    })
    # slightly predictable target (noisy)
    y = pd.Series((X["feat_1"] + np.random.randn(n) * 2 > 0).astype(int))
    prices = pd.Series(close)

    return X, y, prices


class TestWalkForward:
    def test_returns_result_object(self, wf_dataset):
        X, y, prices = wf_dataset
        result = walk_forward_backtest(
            X, y, prices, n_folds=3, monte_carlo_n=0, verbose=False
        )
        assert isinstance(result, WalkForwardResult)

    def test_correct_number_of_folds(self, wf_dataset):
        X, y, prices = wf_dataset
        result = walk_forward_backtest(
            X, y, prices, n_folds=3, monte_carlo_n=0, verbose=False
        )
        assert len(result.folds) == 3

    def test_aggregate_equity_exists(self, wf_dataset):
        X, y, prices = wf_dataset
        result = walk_forward_backtest(
            X, y, prices, n_folds=3, monte_carlo_n=0, verbose=False
        )
        assert result.aggregate_equity is not None
        assert len(result.aggregate_equity) > 0

    def test_summary_has_required_keys(self, wf_dataset):
        X, y, prices = wf_dataset
        result = walk_forward_backtest(
            X, y, prices, n_folds=3, monte_carlo_n=0, verbose=False
        )
        summary = result.summary()
        required = [
            "n_folds", "mean_accuracy", "mean_f1", "mean_sharpe",
            "mean_return", "total_trades",
        ]
        for key in required:
            assert key in summary, f"Missing key: {key}"

    def test_monte_carlo_produces_pvalue(self, wf_dataset):
        X, y, prices = wf_dataset
        result = walk_forward_backtest(
            X, y, prices, n_folds=3, monte_carlo_n=100, verbose=False
        )
        assert result.monte_carlo is not None
        assert "p_value" in result.monte_carlo
        assert 0 <= result.monte_carlo["p_value"] <= 1

    def test_fold_metrics_bounded(self, wf_dataset):
        X, y, prices = wf_dataset
        result = walk_forward_backtest(
            X, y, prices, n_folds=3, monte_carlo_n=0, verbose=False
        )
        for fold in result.folds:
            assert 0 <= fold.accuracy <= 1
            assert 0 <= fold.max_drawdown <= 1
            assert fold.n_train > 0
            assert fold.n_test > 0


class TestHelpers:
    def test_sharpe_positive_returns(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.005, 1000))
        assert _sharpe(returns) > 0

    def test_sharpe_zero_returns(self):
        returns = pd.Series([0.0] * 100)
        assert _sharpe(returns) == 0.0

    def test_max_drawdown_range(self):
        equity = pd.Series([1.0, 1.1, 1.05, 0.9, 0.95, 1.2])
        dd = _max_drawdown(equity)
        assert 0 < dd < 1

    def test_strategy_returns_with_costs(self):
        prices = pd.Series([100, 101, 102, 101, 100])
        y_pred = np.array([1, 1, 0, 0, 1])
        y_true = np.array([1, 1, 0, 0, 1])
        strat_ret, trades = _compute_strategy_returns(y_true, y_pred, prices, cost_bps=10)
        assert len(strat_ret) == len(prices)
        # costs should reduce returns
        strat_no_cost, _ = _compute_strategy_returns(y_true, y_pred, prices, cost_bps=0)
        assert strat_ret.sum() <= strat_no_cost.sum()

    def test_monte_carlo_returns_dict(self):
        returns = pd.Series(np.random.normal(0.001, 0.01, 200))
        result = _monte_carlo_test(returns, n_simulations=50)
        assert "strategy_sharpe" in result
        assert "p_value" in result
        assert "ci_lower" in result
