"""
Phase 2 – Step 1  |  Hybrid Momentum + Mean Reversion Strategy (Crypto)
Author: Om | Strategy Research Lab
"""

# ==============================================================
# Imports
# ==============================================================
import ssl
import yfinance as yf
import datetime as dt
import pandas as pd
from src.indicators import compute_indicators
from src.visualization import plot_dashboard
from src.feature_factory_v2 import build_features
from src.feature_engineering import build_feature_matrix  # legacy Phase 2 only
from src.backtesting import backtest_strategy
from src.ml_signal_engine import train_ml_model


# Bypass Windows/McAfee SSL issues
ssl._create_default_https_context = ssl._create_unverified_context
print("🚀 main.py started successfully")


# ==============================================================
# Data Fetcher
# ==============================================================
def fetch_data(ticker: str, interval='1d', start='2018-01-01', end=None):
    """Fetch OHLCV data using yfinance (Crypto ready)."""
    if end is None:
        end = dt.datetime.now().strftime('%Y-%m-%d')

    print(f"\n📈 Fetching {interval} data for {ticker} ...")
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            print(f"⚠️ No data found for '{ticker}'.")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df.set_index('Date', inplace=True)
        print(f"✅ Fetched {len(df)} rows of data.")
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None


# ==============================================================
# Hybrid Strategy Logic
# ==============================================================
def generate_hybrid_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Mean Reversion (RSI) + Momentum (SMA/EMA)."""
    print("⚙️ Generating hybrid strategy signals...")

    df['Position'] = 0
    df.loc[(df['RSI'] < 30) & (df['SMA_20'] > df['EMA_20']), 'Position'] = 1
    df.loc[(df['RSI'] > 70) | (df['SMA_20'] < df['EMA_20']), 'Position'] = 0
    df['Position'] = df['Position'].shift(1).fillna(0)
    df['Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Position'] * df['Return']
    df['Equity_Curve'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()

    print("✅ Signals + equity curve generated.")
    return df


# ==============================================================
# Performance Summary
# ==============================================================
def performance_summary(df: pd.DataFrame):
    print("\n📊 Performance Summary:")
    if df['Strategy_Return'].std() == 0:
        print("⚠️ No variability in returns.")
        return
    CAGR = (df['Equity_Curve'].iloc[-1])**(252/len(df)) - 1
    Sharpe = df['Strategy_Return'].mean()/df['Strategy_Return'].std() * (252**0.5)
    MaxDD = ((df['Equity_Curve'].cummax()-df['Equity_Curve'])/df['Equity_Curve'].cummax()).max()
    print(f"  CAGR: {CAGR*100:.2f}%\n  Sharpe: {Sharpe:.2f}\n  Max DD: {MaxDD*100:.2f}%")
    return {'CAGR': CAGR, 'Sharpe': Sharpe, 'MaxDD': MaxDD}


# ==============================================================
# Step 1: Hybrid Quant Model
# ==============================================================
def run_step1_hybrid():
    print("\n📊 Welcome to Strategy Research Lab – Phase 2 Hybrid Quant Model")
    ticker = input("Enter crypto ticker (default BTC-USD): ").strip().upper() or "BTC-USD"
    print(f"\n🔹 Using ticker: {ticker}")

    df = fetch_data(ticker)
    if df is None:
        return
    df = compute_indicators(df)
    df = generate_hybrid_signals(df)
    _ = performance_summary(df)
    plot_dashboard(df, ticker)

    out = f"results_{ticker}_phase2_step1.csv"
    df.to_csv(out)
    print(f"\n💾 Saved results to {out}")
    print("\n✅ Phase 2 – Step 1 complete! Ready for Step 2 (Data Engineering).")


# ==============================================================
# Step 2: Feature Engineering
# ==============================================================
def run_step2_feature_engineering():
    print("\n🚀 Starting Phase 2 – Step 2: Data Engineering & Feature Building")

    file_path = "results_BTC-USD_phase2_step1.csv"
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df = build_feature_matrix(df)

    out_path = "phase2_features_BTC.csv"
    df.to_csv(out_path)
    print(f"\n💾 Saved engineered dataset to {out_path}")
    print("✅ Phase 2 – Step 2 complete! Ready for Step 3 (Backtesting / ML Integration).")


# ==============================================================
# Step 3: Backtesting
# ==============================================================
def run_step3_backtest():
    print("\n🚀 Starting Phase 3 – Step 1: Backtesting Engine")

    df = pd.read_csv("phase2_features_BTC.csv", parse_dates=['Date'], index_col='Date')
    metrics = backtest_strategy(df, signal_logic='hybrid')

    print("\n✅ Phase 3 – Step 1 complete! Backtest finished.")
    return metrics


# ==============================================================
# Step 4: ML Signal Engine
# ==============================================================
def run_step4_ml_signals():
    print("\n🚀 Starting Phase 3 – Step 2: Machine Learning Signal Engine")

    df = pd.read_csv("phase2_features_BTC.csv", parse_dates=['Date'], index_col='Date')
    df_ml, model = train_ml_model(df)

    df_ml.to_csv("phase3_ML_signals_BTC.csv")
    print("\n💾 Saved ML-generated signals to phase3_ML_signals_BTC.csv")
    print("✅ Phase 3 – Step 2 complete! Model ready for ML-based backtesting.")
    return df_ml, model


# ==============================================================
# Unified Runner
# ==============================================================
if __name__ == "__main__":
    run_step1_hybrid()              # Step 1
    run_step2_feature_engineering() # Step 2
    run_step3_backtest()            # Step 3
    run_step4_ml_signals()          # Step 4

