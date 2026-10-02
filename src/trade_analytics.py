"""
Trade-Level P&L Analytics Engine
=================================
Parses execution logs and produces institutional-grade trade analytics:
- Per-trade P&L with MAE/MFE
- Rolling performance metrics
- Drawdown analysis
- Win/loss streaks
- Performance by regime, time-of-day, signal confidence
Author: Om | Strategy Research Lab
"""

import numpy as np
import pandas as pd
import json
import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("trade_analytics")


# ------------------------------------------------------------------
# Trade extraction from executor logs
# ------------------------------------------------------------------
def extract_trades_from_log(log_path: str) -> pd.DataFrame:
    """
    Parse executor CSV log and extract round-trip trades.
    Each BUY opens a trade; the next SELL closes it.
    Returns DataFrame with one row per closed trade.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    df = pd.read_csv(log_path, parse_dates=["timestamp"])

    # normalize column names
    df.columns = df.columns.str.strip().str.lower()

    if "action" not in df.columns:
        raise ValueError("Log must have an 'action' column (BUY/SELL/HOLD)")

    trades = []
    open_trade = None

    for _, row in df.iterrows():
        action = str(row.get("action", "")).upper().strip()

        if action == "BUY" and open_trade is None:
            open_trade = {
                "entry_time": row["timestamp"],
                "entry_price": float(row.get("entry_price") or row.get("close", 0)),
                "qty": float(row.get("qty", 0)),
                "entry_signal_prob": float(row.get("signal_prob", 0))
                    if "signal_prob" in row.index else None,
                "candles_during": [float(row.get("close", 0))],
            }

        elif action in ("SELL",) and open_trade is not None:
            exit_price = float(row.get("exit_price") or row.get("close", 0))
            entry_price = open_trade["entry_price"]
            qty = open_trade["qty"]

            # MAE / MFE from candles during the trade
            candles = open_trade["candles_during"]
            diffs = [p - entry_price for p in candles]
            mae = min(diffs) if diffs else 0.0
            mfe = max(diffs) if diffs else 0.0

            pnl = (exit_price - entry_price) * qty
            pnl_pct = (exit_price / entry_price - 1.0) * 100 if entry_price > 0 else 0.0
            holding_time = (row["timestamp"] - open_trade["entry_time"]).total_seconds()

            trades.append({
                "entry_time": open_trade["entry_time"],
                "exit_time": row["timestamp"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "qty": qty,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "mae": mae,
                "mfe": mfe,
                "holding_seconds": holding_time,
                "entry_signal_prob": open_trade["entry_signal_prob"],
            })
            open_trade = None

        elif open_trade is not None:
            # accumulate candle prices for MAE/MFE
            close_val = row.get("close", None)
            if close_val is not None and not pd.isna(close_val):
                open_trade["candles_during"].append(float(close_val))

    return pd.DataFrame(trades)


# ------------------------------------------------------------------
# Core analytics
# ------------------------------------------------------------------
def compute_trade_metrics(trades_df: pd.DataFrame) -> dict:
    """Compute aggregate metrics from a trades DataFrame."""
    if trades_df.empty:
        return {"error": "no trades"}

    n = len(trades_df)
    winners = trades_df[trades_df["pnl"] > 0]
    losers = trades_df[trades_df["pnl"] < 0]
    flat = trades_df[trades_df["pnl"] == 0]

    total_pnl = trades_df["pnl"].sum()
    win_rate = len(winners) / n if n > 0 else 0
    avg_win = winners["pnl"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl"].mean() if len(losers) > 0 else 0
    profit_factor = abs(winners["pnl"].sum() / losers["pnl"].sum()) \
        if len(losers) > 0 and losers["pnl"].sum() != 0 else float("inf")

    # expectancy per trade
    expectancy = total_pnl / n if n > 0 else 0

    # max consecutive wins/losses
    streaks = _compute_streaks(trades_df["pnl"].values)

    # cumulative equity for drawdown
    cum_pnl = trades_df["pnl"].cumsum()
    peak = cum_pnl.cummax()
    drawdown = peak - cum_pnl
    max_dd = drawdown.max()

    # holding time stats
    avg_hold = trades_df["holding_seconds"].mean()
    median_hold = trades_df["holding_seconds"].median()

    # MAE/MFE
    avg_mae = trades_df["mae"].mean()
    avg_mfe = trades_df["mfe"].mean()

    return {
        "total_trades": n,
        "winners": len(winners),
        "losers": len(losers),
        "flat": len(flat),
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 4),
        "avg_pnl_per_trade": round(expectancy, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(max_dd, 4),
        "max_consecutive_wins": streaks["max_wins"],
        "max_consecutive_losses": streaks["max_losses"],
        "avg_holding_seconds": round(avg_hold, 1),
        "median_holding_seconds": round(median_hold, 1),
        "avg_mae": round(avg_mae, 6),
        "avg_mfe": round(avg_mfe, 6),
    }


def _compute_streaks(pnl_array):
    """Compute max consecutive wins and losses."""
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for pnl in pnl_array:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
        else:
            current_wins = 0
            current_losses = 0
        max_wins = max(max_wins, current_wins)
        max_losses = max(max_losses, current_losses)

    return {"max_wins": max_wins, "max_losses": max_losses}


# ------------------------------------------------------------------
# Rolling performance (for dashboards)
# ------------------------------------------------------------------
def rolling_metrics(trades_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute rolling metrics over the last N trades."""
    if trades_df.empty or len(trades_df) < window:
        return pd.DataFrame()

    records = []
    for i in range(window, len(trades_df) + 1):
        chunk = trades_df.iloc[i - window:i]
        wins = (chunk["pnl"] > 0).sum()
        records.append({
            "trade_num": i,
            "exit_time": chunk.iloc[-1]["exit_time"],
            "rolling_pnl": chunk["pnl"].sum(),
            "rolling_win_rate": wins / window,
            "rolling_avg_pnl": chunk["pnl"].mean(),
            "rolling_sharpe": chunk["pnl_pct"].mean() / (chunk["pnl_pct"].std() + 1e-9),
        })

    return pd.DataFrame(records)


# ------------------------------------------------------------------
# Performance by signal confidence bucket
# ------------------------------------------------------------------
def performance_by_confidence(trades_df: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    """Break down win rate and avg PnL by entry signal probability."""
    if trades_df.empty or "entry_signal_prob" not in trades_df.columns:
        return pd.DataFrame()

    df = trades_df.dropna(subset=["entry_signal_prob"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["conf_bucket"] = pd.qcut(
        df["entry_signal_prob"], q=n_buckets, duplicates="drop"
    )

    grouped = df.groupby("conf_bucket", observed=True).agg(
        n_trades=("pnl", "count"),
        win_rate=("pnl", lambda x: (x > 0).mean()),
        avg_pnl=("pnl", "mean"),
        total_pnl=("pnl", "sum"),
        avg_pnl_pct=("pnl_pct", "mean"),
    ).reset_index()

    return grouped


# ------------------------------------------------------------------
# Full report runner
# ------------------------------------------------------------------
def generate_report(
    log_path: str = "executor_v4_log.csv",
    output_path: str = "data/trade_analytics_report.json",
) -> dict:
    """Generate full trade analytics report from executor log."""
    logging.basicConfig(level=logging.INFO)

    print(f"Extracting trades from {log_path}...")
    trades_df = extract_trades_from_log(log_path)

    if trades_df.empty:
        print("No closed trades found in log.")
        return {"error": "no closed trades"}

    print(f"Found {len(trades_df)} closed trades")

    metrics = compute_trade_metrics(trades_df)
    rolling = rolling_metrics(trades_df)
    by_conf = performance_by_confidence(trades_df)

    report = {
        "source": log_path,
        "metrics": metrics,
        "rolling_summary": {
            "last_20_pnl": float(rolling.iloc[-1]["rolling_pnl"]) if not rolling.empty else None,
            "last_20_win_rate": float(rolling.iloc[-1]["rolling_win_rate"]) if not rolling.empty else None,
        },
        "by_confidence": by_conf.to_dict(orient="records") if not by_conf.empty else [],
    }

    # print summary
    print("\n" + "=" * 55)
    print("TRADE ANALYTICS REPORT")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<28}: {v}")
    print("=" * 55)

    # save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {output_path}")

    # save trades CSV
    trades_csv = output_path.replace(".json", "_trades.csv")
    trades_df.to_csv(trades_csv, index=False)
    print(f"Trades saved to {trades_csv}")

    return report


if __name__ == "__main__":
    generate_report()
