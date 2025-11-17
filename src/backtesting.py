"""
Phase 3 – Step 1 | Backtesting & Performance Evaluation Engine
Evaluates strategy or ML signals using feature dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------
def calculate_performance(df: pd.DataFrame, signal_col='Signal', ret_col='Return'):
    """Compute key performance metrics."""
    df['Strategy_Return'] = df[signal_col].shift(1) * df[ret_col]
    df['Equity_Curve'] = (1 + df['Strategy_Return']).cumprod()
    df['Benchmark_Curve'] = (1 + df[ret_col]).cumprod()

    total_days = len(df)
    total_years = total_days / 252
    cagr = (df['Equity_Curve'].iloc[-1]) ** (1 / total_years) - 1
    sharpe = np.sqrt(252) * df['Strategy_Return'].mean() / (df['Strategy_Return'].std() + 1e-9)
    sortino = np.sqrt(252) * df['Strategy_Return'].mean() / (
        df.loc[df['Strategy_Return'] < 0, 'Strategy_Return'].std() + 1e-9)
    max_dd = ((df['Equity_Curve'].cummax() - df['Equity_Curve']) / df['Equity_Curve'].cummax()).max()

    return {
        'CAGR': cagr, 'Sharpe': sharpe, 'Sortino': sortino,
        'MaxDD': max_dd, 'Final_Equity': df['Equity_Curve'].iloc[-1]
    }

# --------------------------------------------------------------
def backtest_strategy(df: pd.DataFrame, signal_logic='hybrid'):
    """
    Backtest using hybrid or ML-based signals.
    signal_logic = 'hybrid' → RSI & SMA/EMA rules
    signal_logic = 'ml' → uses Target / ML_Pred column
    """

    print("🧮 Running backtest...")

    # Calculate simple daily returns
    df['Return'] = df['Close'].pct_change().fillna(0)

    if signal_logic == 'hybrid':
        df['Signal'] = 0
        df.loc[(df['RSI'] < 30) & (df['SMA_20'] > df['EMA_20']), 'Signal'] = 1
        df.loc[(df['RSI'] > 70) | (df['SMA_20'] < df['EMA_20']), 'Signal'] = 0
        df['Signal'] = df['Signal'].ffill().fillna(0)
    elif signal_logic == 'ml':
        if 'Predicted_Label' in df.columns:
            df['Signal'] = df['Predicted_Label']
        elif 'Target' in df.columns:
            df['Signal'] = df['Target']
        else:
            raise ValueError("No ML prediction or Target column found.")
    else:
        raise ValueError("signal_logic must be 'hybrid' or 'ml'.")

    # Compute metrics
    metrics = calculate_performance(df)
    print("\n📊 Performance Report:")
    for k, v in metrics.items():
        print(f"  {k:<15}: {v*100:.2f}%" if k != 'Final_Equity' else f"  {k:<15}: {v:.2f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Equity_Curve'], label='Strategy Equity')
    plt.plot(df.index, df['Benchmark_Curve'], label='Buy & Hold', alpha=0.6)
    plt.title("Equity Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    return metrics
