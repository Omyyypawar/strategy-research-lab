"""
dataset_builder_v2.py
Builds feature dataset using Feature Factory V2 and creates a 5-minute binary target.
Saves: data/features_v2.parquet
"""
import os
import pandas as pd
from dotenv import load_dotenv
from feature_factory_v2 import build_features
from binance.client import Client

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "1m")
OUT_PATH = "data/features_v2.parquet"

def fetch_historical_klines(symbol, interval="1m", limit=2000):
    client = Client(API_KEY, API_SECRET)
    data = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","tbbav","tbqav","ignore"
    ])
    df = df[["open","high","low","close","volume","close_time"]]
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df

def build_dataset(df):
    # build features
    full = build_features(df)
    # create 5-minute future return target (5 x 1m candles = 5)
    full["future_close_5m"] = full["close"].shift(-5)
    full["future_return_5m"] = (full["future_close_5m"] / full["close"]) - 1
    full["target"] = (full["future_return_5m"] > 0).astype(int)

    # drop rows with NaNs introduced by shift/rolling
    full = full.dropna().reset_index(drop=True)

    # keep timestamp if exists
    if "close_time" in full.columns:
        full.rename(columns={"close_time": "timestamp"}, inplace=True)
    return full

def main():
    print("Fetching historical klines...")
    df = fetch_historical_klines(SYMBOL, INTERVAL, limit=2000)
    print("Building features (V2)...")
    dataset = build_dataset(df)
    os.makedirs("data", exist_ok=True)
    dataset.to_parquet(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)
    print("Shape:", dataset.shape)

if __name__ == "__main__":
    main()
