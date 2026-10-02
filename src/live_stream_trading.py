import os
import pandas as pd
import numpy as np
import datetime as dt
import joblib
import asyncio
import json
from collections import deque
from dotenv import load_dotenv
from binance import AsyncClient, BinanceSocketManager
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
TICKER = "BTCUSDT"
INTERVAL = "1m"
LOOKBACK = 100
MODEL_PATH = "phase3_rf_model.pkl"
START_BALANCE = 1000

# Load environment
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
USE_TESTNET = os.getenv("BINANCE_TESTNET", "True").lower() == "true"

# --------------------------------------------------
# Feature Calculations
# --------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def compute_features(df):
    df["Return"] = df["Close"].pct_change()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD_Line"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(df["Close"])
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["EMA_5"] = df["Close"].ewm(span=5).mean()
    df["EMA_Ratio"] = df["EMA_5"] / (df["EMA_20"] + 1e-9)
    for n in [1, 2, 3, 5]:
        df[f"Lag_{n}"] = df["Close"].shift(n)
    df.dropna(inplace=True)
    return df

# --------------------------------------------------
# Live WebSocket Paper Trading
# --------------------------------------------------
async def main():
    client = await AsyncClient.create(API_KEY, API_SECRET, testnet=USE_TESTNET)
    bm = BinanceSocketManager(client)
    ts = bm.kline_socket(TICKER, interval=INTERVAL)

    model = joblib.load(MODEL_PATH)
    balance = START_BALANCE
    position = 0.0
    equity_curve = []
    timestamps = deque(maxlen=500)
    price_data = deque(maxlen=LOOKBACK)

    print(f"🚀 Starting REAL-TIME WebSocket Paper Trading Engine for {TICKER}")

    async with ts as stream:
        while True:
            res = await stream.recv()
            candle = res["k"]
            close_price = float(candle["c"])
            close_time = dt.datetime.fromtimestamp(candle["T"] / 1000)

            price_data.append(close_price)
            if len(price_data) < LOOKBACK:
                continue

            df = pd.DataFrame({"Close": list(price_data)})
            df["Date"] = pd.date_range(end=close_time, periods=len(df), freq="T")
            df = compute_features(df)

            features = ["Close", "SMA_20", "EMA_20", "RSI", "MACD_Line", "EMA_Ratio", "Lag_1", "Lag_2", "Lag_3", "Lag_5"]
            X_live = df.iloc[-1:][features]
            try:
                signal = int(model.predict(X_live)[0])
            except Exception as e:
                print(f"Prediction error: {e}")
                continue

            # Trade simulation
            if signal == 1 and balance > 0:
                position = balance / close_price
                balance = 0
                action = "BUY"
            elif signal == 0 and position > 0:
                balance = position * close_price
                position = 0
                action = "SELL"
            else:
                action = "HOLD"

            total_equity = balance + position * close_price
            equity_curve.append(total_equity)
            timestamps.append(close_time)

            print(f"[{close_time.strftime('%H:%M:%S')}] {action:<5} | Price: {close_price:.2f} | Equity: {total_equity:.2f}")

            if len(equity_curve) % 10 == 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(timestamps), y=list(equity_curve), mode="lines", name="Equity"))
                fig.update_layout(template="plotly_dark", title="Live WebSocket Paper Trading", height=500)
                fig.show()

    await client.close_connection()

if __name__ == "__main__":
    asyncio.run(main())
