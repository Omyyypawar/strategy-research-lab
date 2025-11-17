# src/live_trading.py
import pandas as pd
import numpy as np
import requests
import time
import datetime as dt
import joblib
import plotly.graph_objects as go
import plotly.io as pio
import os
from collections import deque

pio.renderers.default = "browser"

# -----------------------------
# CONFIG
# -----------------------------
TICKER = "BTCUSDT"
INTERVAL = "1m"
LOOKBACK = 200          # increase a bit for indicator windows
MODEL_PATH = "phase3_rf_model.pkl"
START_BALANCE = 1000
REFRESH_SECS = 30
LOG_FILE = "paper_trading_log.csv"

# -----------------------------
# Binance public klines
# -----------------------------
# --------------------------------------------------
# Binance Live Connection (via Testnet)
# --------------------------------------------------
from binance.client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
USE_TESTNET = os.getenv("BINANCE_TESTNET", "True").lower() == "true"

# Initialize Binance client
client = Client(API_KEY, API_SECRET, testnet=USE_TESTNET)

def get_recent_klines(symbol=TICKER, interval=INTERVAL, limit=LOOKBACK):
    """Fetch live candlestick data directly from Binance (supports Testnet)."""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            "OpenTime", "Open", "High", "Low", "Close", "Volume",
            "CloseTime", "QuoteAssetVol", "Trades", "TBBAV", "TBQAV", "Ignore"
        ])
        df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms")
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Keep consistent with training data
        df = df[["OpenTime", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.rename(columns={"OpenTime": "Date"}, inplace=True)
        df.set_index("Date", inplace=True)

        return df

    except Exception as e:
        print(f"⚠️ Binance data fetch failed: {e}")
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])


# -----------------------------
# indicator helpers
# -----------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

# -----------------------------
# features (must match training)
# -----------------------------
def compute_features(df):
    """Create features to match training-time features. Operates inplace and returns df."""
    # Basic returns
    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1)).replace([np.inf, -np.inf], 0).fillna(0)

    # Moving averages
    df["SMA_20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA_5"] = df["Close"].rolling(5, min_periods=1).mean()
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["SMA_Ratio"] = df["SMA_5"] / (df["SMA_20"] + 1e-9)
    df["EMA_Ratio"] = df["EMA_5"] / (df["EMA_20"] + 1e-9)

    # RSI
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["RSI_Change"] = df["RSI"].diff().fillna(0)
    df["RSI_Smooth"] = df["RSI"].rolling(5, min_periods=1).mean()

    # MACD
    df["MACD_Line"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(df["Close"])

    # Bollinger Bands
    df["BB_Mid"] = df["SMA_20"]
    std20 = df["Close"].rolling(20, min_periods=1).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * std20
    df["BB_Lower"] = df["BB_Mid"] - 2 * std20
    denom = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    df["BB_Signal"] = ((df["Close"] - df["BB_Mid"]) / (denom + 1e-9)).fillna(0)

    # Momentum & Volatility
    for n in [3, 7, 14]:
        df[f"Momentum_{n}"] = df["Close"] - df["Close"].shift(n)
    for n in [5, 10]:
        df[f"Volatility_{n}"] = df["Return"].rolling(n, min_periods=1).std().fillna(0)

    # Lags
    for n in [1, 2, 3, 5]:
        df[f"Lag_{n}"] = df["Close"].shift(n)

    # SmartMom (custom): combination of momentum & MACD & RSI change
    df["SmartMom"] = (df.get("Momentum_7", 0).fillna(0) + df["MACD_Line"].fillna(0) + df["RSI_Change"].fillna(0)).fillna(0)

    # Position proxy (simple direction)
    df["Position"] = np.where(df["Return"] > 0, 1.0, -1.0)

    # Future_Return used at training as label — set to 0 in live inference
    df["Future_Return"] = 0.0

    # If any NaNs remain, drop earliest rows; keep enough rows for lookbacks
    df.dropna(inplace=True)
    return df

# -----------------------------
# helper: align X to model features
# -----------------------------
def build_model_input(df_latest, model, fallback_phase2_csv="phase2_features_BTC.csv"):
    """
    Ensure X has exactly the same columns the model expects (and in the same order).
    Missing columns will be added (filled with 0). If the model doesn't provide feature_names_in_,
    try reading names from fallback_phase2_csv numeric headers.
    """
    # try to get expected names from model
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        expected = None

    # fallback: try reading from phase2 CSV header (common pattern)
    if expected is None and os.path.exists(fallback_phase2_csv):
        try:
            df_phase2 = pd.read_csv(fallback_phase2_csv, nrows=1)
            expected = [c for c in df_phase2.columns if c not in ("Target", "Date")]
        except Exception:
            expected = None

    # final fallback: use numeric columns from df_latest
    if expected is None:
        expected = [c for c in df_latest.columns if np.issubdtype(df_latest[c].dtype, np.number)]

    # create X with expected columns in order
    X = pd.DataFrame(index=df_latest.index)
    missing = []
    for c in expected:
        if c in df_latest.columns:
            X[c] = df_latest[c].astype(float)
        else:
            X[c] = 0.0
            missing.append(c)

    # reorder to expected (should already be)
    X = X[expected]

    return X, expected, missing

# -----------------------------
# main live loop
# -----------------------------
def run_paper_trading():
    print(f"🚀 Starting Live Paper Trading Engine for {TICKER}")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    balance = START_BALANCE
    position = 0.0
    equity_curve = []
    timestamps = deque(maxlen=500)

    # prepare log file
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=["timestamp", "price", "action", "signal", "balance", "position", "equity"]).to_csv(LOG_FILE, index=False)

    loop_count = 0
    last_missing_reported = None

    while True:
        loop_count += 1
        try:
            df = get_recent_klines()
            df = compute_features(df)   # compute all features, use index=date

            latest = df.iloc[-1:]

            # build X aligned to model features
            X_live, expected_features, missing = build_model_input(latest, model)

            # report missing columns (but only once per distinct set)
            missing_key = tuple(missing)
            if missing and missing_key != last_missing_reported:
                print(f"Warning: {len(missing)} expected model columns were not found in live features. They will be filled with 0: {missing[:10]}{'...' if len(missing)>10 else ''}")
                last_missing_reported = missing_key

            # make prediction (ensure shape is 2D)
            try:
                preds = model.predict(X_live)
                signal = int(preds[0])
            except Exception as mp_err:
                # try predict_proba fallback for probabilistic models
                try:
                    probs = model.predict_proba(X_live)
                    signal = int((probs[:, 1] > 0.5)[0])
                except Exception:
                    raise mp_err

            price = float(latest["Close"].values[0])

            # trading simulation
            action = "HOLD"
            if signal == 1 and balance > 0:
                position = balance / price
                balance = 0.0
                action = "BUY"
            elif signal == 0 and position > 0:
                balance = position * price
                position = 0.0
                action = "SELL"

            total_equity = balance + position * price
            equity_curve.append(total_equity)
            timestamps.append(latest.index[0])

            # log to CSV
            new_row = pd.DataFrame([{
                "timestamp": dt.datetime.now().isoformat(),
                "price": price,
                "action": action,
                "signal": int(signal),
                "balance": balance,
                "position": position,
                "equity": total_equity
            }])
            new_row.to_csv(LOG_FILE, mode="a", header=False, index=False)

            print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {action:<5} | Price: {price:.2f} | Equity: {total_equity:.2f}")

            # show plot every 10 iterations
            if len(equity_curve) % 10 == 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(timestamps), y=list(equity_curve), mode="lines", name="Equity"))
                fig.update_layout(template="plotly_dark", title="Live Paper Trading Equity Curve", height=500)
                fig.show()

        except Exception as e:
            # print a compact, readable error
            print(f"Model prediction error: {e}")

        time.sleep(REFRESH_SECS)

if __name__ == "__main__":
    run_paper_trading()
