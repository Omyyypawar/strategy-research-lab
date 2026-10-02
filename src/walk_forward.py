"""
Walk-Forward Backtesting Engine with Monte Carlo Confidence Intervals
=====================================================================
Proper out-of-sample validation for ML trading strategies.
Author: Om | Strategy Research Lab

Key features:
- Anchored and rolling walk-forward splits
- Per-fold model retraining (no look-ahead bias)
- Transaction cost & slippage modeling
- Monte Carlo permutation for confidence intervals
- Comprehensive per-fold and aggregate metrics
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger("walk_forward")


# ------------------------------------------------------------------
# Data classes for results
# ------------------------------------------------------------------
@dataclass
class FoldResult:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    n_train: int
    n_test: int
    accuracy: float
    f1: float
    precision: float
    recall: float
    total_return: float
    sharpe: float
    max_drawdown: float
    n_trades: int
    win_rate: float


@dataclass
class WalkForwardResult:
    folds: list = field(default_factory=list)
    aggregate_equity: Optional[pd.Series] = None
    aggregate_returns: Optional[pd.Series] = None
    monte_carlo: Optional[dict] = None

    def summary(self) -> dict:
        if not self.folds:
            return {}
        return {
            "n_folds": len(self.folds),
            "mean_accuracy": np.mean([f.accuracy for f in self.folds]),
            "mean_f1": np.mean([f.f1 for f in self.folds]),
            "mean_sharpe": np.mean([f.sharpe for f in self.folds]),
            "mean_return": np.mean([f.total_return for f in self.folds]),
            "mean_max_dd": np.mean([f.max_drawdown for f in self.folds]),
            "mean_win_rate": np.mean([f.win_rate for f in self.folds]),
            "total_trades": sum(f.n_trades for f in self.folds),
            "worst_fold_return": min(f.total_return for f in self.folds),
            "best_fold_return": max(f.total_return for f in self.folds),
        }

    def print_summary(self):
        s = self.summary()
        print("\n" + "=" * 60)
        print("WALK-FORWARD BACKTEST RESULTS")
        print("=" * 60)
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k:<25}: {v:>10.4f}")
            else:
                print(f"  {k:<25}: {v:>10}")
        if self.monte_carlo:
            mc = self.monte_carlo
            print(f"\n  Monte Carlo ({mc['n_simulations']} permutations):")
            print(f"    Strategy Sharpe       : {mc['strategy_sharpe']:>10.4f}")
            print(f"    Random mean Sharpe    : {mc['random_mean_sharpe']:>10.4f}")
            print(f"    p-value               : {mc['p_value']:>10.4f}")
            print(f"    95% CI (Sharpe)       : [{mc['ci_lower']:.4f}, {mc['ci_upper']:.4f}]")
            print(f"    Significant (p<0.05)  : {'YES' if mc['p_value'] < 0.05 else 'NO'}")
        print("=" * 60)


# ------------------------------------------------------------------
# Performance helpers
# ------------------------------------------------------------------
def _compute_strategy_returns(y_true, y_pred, prices, cost_bps=5.0):
    """
    Compute strategy returns from predictions.
    - y_pred=1 means go long, y_pred=0 means flat
    - Applies transaction cost on position changes
    """
    position = pd.Series(y_pred, index=prices.index)
    price_returns = prices.pct_change().fillna(0.0)

    # strategy returns = position(t-1) * return(t)
    strat_returns = position.shift(1).fillna(0) * price_returns

    # transaction costs on position changes
    trades = position.diff().abs().fillna(0)
    costs = trades * (cost_bps / 10000.0)
    strat_returns = strat_returns - costs

    return strat_returns, trades


def _sharpe(returns, periods_per_year=252 * 24 * 60):
    """Annualized Sharpe ratio. Default assumes 1-min bars."""
    if returns.std() < 1e-12:
        return 0.0
    return np.sqrt(periods_per_year) * returns.mean() / returns.std()


def _max_drawdown(equity):
    peak = equity.cummax()
    dd = (peak - equity) / (peak + 1e-12)
    return dd.max()


def _win_rate(returns):
    trades = returns[returns != 0]
    if len(trades) == 0:
        return 0.0
    return (trades > 0).sum() / len(trades)


# ------------------------------------------------------------------
# Walk-forward engine
# ------------------------------------------------------------------
def walk_forward_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    prices: pd.Series,
    n_folds: int = 5,
    train_ratio: float = 0.7,
    anchored: bool = True,
    model_factory=None,
    cost_bps: float = 5.0,
    monte_carlo_n: int = 1000,
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Walk-forward backtest with expanding or rolling training window.

    Parameters
    ----------
    X : DataFrame of features (aligned index with y and prices)
    y : Series of binary targets
    prices : Series of close prices (same index as X)
    n_folds : number of test folds
    train_ratio : fraction of data for initial training window
    anchored : if True, training window grows (anchored); if False, rolling
    model_factory : callable returning a fresh model; defaults to RandomForest
    cost_bps : round-trip transaction cost in basis points
    monte_carlo_n : number of permutation simulations (0 to skip)
    verbose : print per-fold results

    Returns
    -------
    WalkForwardResult with per-fold metrics and aggregate equity curve
    """
    if model_factory is None:
        model_factory = lambda: RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )

    n = len(X)
    initial_train = int(n * train_ratio)
    test_size = (n - initial_train) // n_folds

    if test_size < 10:
        raise ValueError(
            f"Test fold too small ({test_size} rows). Need more data or fewer folds."
        )

    result = WalkForwardResult()
    all_test_returns = []
    all_test_indices = []

    for fold_i in range(n_folds):
        # define split boundaries
        test_start = initial_train + fold_i * test_size
        test_end = min(test_start + test_size, n)

        if anchored:
            train_start = 0
        else:
            train_start = max(0, test_start - initial_train)
        train_end = test_start

        if test_end <= test_start:
            break

        # split data
        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]
        prices_test = prices.iloc[test_start:test_end]

        # train fresh model
        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # classification metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        # strategy returns
        strat_ret, trades = _compute_strategy_returns(
            y_test.values, y_pred, prices_test, cost_bps=cost_bps
        )
        equity = (1 + strat_ret).cumprod()
        total_ret = equity.iloc[-1] - 1.0
        sharpe = _sharpe(strat_ret)
        max_dd = _max_drawdown(equity)
        n_trades = int(trades.sum())
        wr = _win_rate(strat_ret)

        fold_result = FoldResult(
            fold=fold_i + 1,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            n_train=len(X_train),
            n_test=len(X_test),
            accuracy=acc,
            f1=f1,
            precision=prec,
            recall=rec,
            total_return=total_ret,
            sharpe=sharpe,
            max_drawdown=max_dd,
            n_trades=n_trades,
            win_rate=wr,
        )
        result.folds.append(fold_result)
        all_test_returns.append(strat_ret)
        all_test_indices.append(strat_ret.index)

        if verbose:
            logger.info(
                "Fold %d/%d | Train=%d Test=%d | Acc=%.3f F1=%.3f | "
                "Return=%.4f Sharpe=%.2f DD=%.4f Trades=%d WR=%.2f",
                fold_i + 1, n_folds, len(X_train), len(X_test),
                acc, f1, total_ret, sharpe, max_dd, n_trades, wr,
            )

    # aggregate equity curve (concatenate all OOS folds)
    if all_test_returns:
        agg_returns = pd.concat(all_test_returns)
        agg_equity = (1 + agg_returns).cumprod()
        result.aggregate_returns = agg_returns
        result.aggregate_equity = agg_equity

    # Monte Carlo permutation test
    if monte_carlo_n > 0 and result.aggregate_returns is not None:
        result.monte_carlo = _monte_carlo_test(
            result.aggregate_returns, n_simulations=monte_carlo_n
        )

    return result


# ------------------------------------------------------------------
# Monte Carlo permutation test
# ------------------------------------------------------------------
def _monte_carlo_test(strategy_returns: pd.Series, n_simulations: int = 1000) -> dict:
    """
    Test if strategy Sharpe is statistically significant vs random signals.
    Permutes the sign of returns to create null distribution.
    """
    actual_sharpe = _sharpe(strategy_returns)
    n = len(strategy_returns)
    returns_arr = strategy_returns.values

    rng = np.random.default_rng(42)
    random_sharpes = np.zeros(n_simulations)

    for i in range(n_simulations):
        # randomly flip signs (equivalent to random long/short)
        signs = rng.choice([-1, 1], size=n)
        shuffled = returns_arr * signs
        std = shuffled.std()
        if std < 1e-12:
            random_sharpes[i] = 0.0
        else:
            random_sharpes[i] = np.sqrt(252 * 24 * 60) * shuffled.mean() / std

    # p-value: fraction of random sharpes >= actual
    p_value = (random_sharpes >= actual_sharpe).sum() / n_simulations
    ci_lower = np.percentile(random_sharpes, 2.5)
    ci_upper = np.percentile(random_sharpes, 97.5)

    return {
        "strategy_sharpe": actual_sharpe,
        "random_mean_sharpe": random_sharpes.mean(),
        "random_std_sharpe": random_sharpes.std(),
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_simulations": n_simulations,
    }


# ------------------------------------------------------------------
# Convenience runner (loads parquet, runs WF, saves results)
# ------------------------------------------------------------------
def run_walk_forward(
    data_path: str = "data/features_v2.parquet",
    output_path: str = "data/walk_forward_results.json",
    n_folds: int = 5,
    cost_bps: float = 5.0,
    monte_carlo_n: int = 1000,
):
    """End-to-end walk-forward from parquet file."""
    logging.basicConfig(level=logging.INFO)

    print("Loading dataset...")
    df = pd.read_parquet(data_path)
    df = df.drop(columns=["timestamp", "future_close_5m", "future_return_5m"], errors="ignore")
    df = df.dropna()

    prices = df["close"].copy()
    X = df.drop(columns=["target"])
    y = df["target"]

    print(f"Dataset: {len(df)} rows, {X.shape[1]} features")
    print(f"Running {n_folds}-fold walk-forward backtest...")

    result = walk_forward_backtest(
        X, y, prices,
        n_folds=n_folds,
        cost_bps=cost_bps,
        monte_carlo_n=monte_carlo_n,
    )

    result.print_summary()

    # save results
    summary = result.summary()
    if result.monte_carlo:
        summary["monte_carlo"] = result.monte_carlo
    summary["per_fold"] = [
        {
            "fold": f.fold,
            "accuracy": f.accuracy,
            "f1": f.f1,
            "sharpe": f.sharpe,
            "total_return": f.total_return,
            "max_drawdown": f.max_drawdown,
            "n_trades": f.n_trades,
            "win_rate": f.win_rate,
        }
        for f in result.folds
    ]

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return result


if __name__ == "__main__":
    run_walk_forward()
