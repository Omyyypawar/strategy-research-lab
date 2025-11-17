# src/live_executor.py
"""
LIVE EXECUTOR V2 — Strategy Research Lab
-----------------------------------------
This version:
✔ Uses FeatureFactory V2 (same as training)
✔ Produces identical features to dataset_builder_v2
✔ Logs OHLCV + Features + Signal + Action in CSV
✔ Warm-starts using historical klines
✔ Fully compatible with models/phase3_rf_v2.pkl
"""

import os
import asyncio
import joblib
import pandas as pd
import numpy as np
import datetime as dt
from collections import deque
from dotenv import load_dotenv
from binance import AsyncClient, BinanceSocketManager
import json
import math
import time
import logging

# ---------------------------------------------------------
# ENV / CONFIG
# ---------------------------------------------------------
load_dotenv()

API_KEY      = os.getenv("BINANCE_API_KEY", "")
API_SECRET   = os.getenv("BINANCE_API_SECRET", "")
USE_TESTNET  = os.getenv("BINANCE_TESTNET", "True").lower() in ("1","true","yes")

# default to the v2 model path we just trained
MODEL_PATH   = os.getenv("MODEL_PATH", "models/phase3_rf_v2.pkl")
SYMBOL       = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL     = os.getenv("INTERVAL", "1m")

ORDER_COOLDOWN   = int(os.getenv("ORDER_COOLDOWN", "30"))
RISK_PER_TRADE   = float(os.getenv("RISK_PER_TRADE", "0.02"))
START_BALANCE    = float(os.getenv("START_BALANCE", "1000"))
LOG_CSV          = os.getenv("LOG_FILE", "executor_log.csv")

LOOKBACK         = 2000     # enough for all V2 features
WARMUP_KLINES    = 500      # fetch size for initial warm start

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("live_exec_v2")

# ---------------------------------------------------------
# FEATURE FACTORY (V2) - import the training builder
# ---------------------------------------------------------
from feature_factory_v2 import build_features

# ---------------------------------------------------------
# Binance helpers
# ---------------------------------------------------------
async def get_symbol_info(client, symbol):
    return await client.get_symbol_info(symbol)


def quantize_qty(qty, step):
    precision = int(round(-math.log10(step))) if step < 1 else 0
    return math.floor(qty * (10**precision)) / (10**precision)


def calc_order_qty(balance_usd, risk_frac, price, step):
    usd_to_trade = balance_usd * risk_frac
    qty = usd_to_trade / price
    return quantize_qty(qty, step)

# helper to write header with full feature list (safe)
def write_header_if_missing(path, base_cols, feature_cols):
    if not os.path.exists(path):
        cols = base_cols[:]
        # insert feature cols after ohlcv fields to keep order logical
        insert_pos = len(["timestamp","open","high","low","close","volume"])
        for f in feature_cols:
            if f not in cols:
                cols.insert(insert_pos, f)
                insert_pos += 1
        pd.DataFrame(columns=cols).to_csv(path, index=False)


# ---------------------------------------------------------
# MAIN EXECUTION LOOP
# ---------------------------------------------------------
async def main():
    logger.info("Starting Live Executor V2 | Symbol=%s Testnet=%s", SYMBOL, USE_TESTNET)

    client = await AsyncClient.create(API_KEY, API_SECRET, testnet=USE_TESTNET)
    bm = BinanceSocketManager(client)
    ksocket = bm.kline_socket(SYMBOL, interval=INTERVAL)

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Loaded ML model: %s", MODEL_PATH)
    except Exception as e:
        logger.exception("Model load failure: %s", e)
        await client.close_connection()
        return

    # Get trading rules (lot step, min qty)
    info = await get_symbol_info(client, SYMBOL)
    if not info:
        logger.error("Could not fetch Binance symbol info.")
        await client.close_connection()
        return

    lot_step = 1e-8
    min_qty = 0.0
    for f in info.get("filters", []):
        if f.get("filterType") == "LOT_SIZE":
            lot_step = float(f.get("stepSize", lot_step))
            min_qty = float(f.get("minQty", 0.0))

    # Rolling OHLCV buffer
    ohlcv = deque(maxlen=LOOKBACK)

    # Prepare base columns (we will extend header after warm-up)
    base_cols = [
        "timestamp","open","high","low","close","volume",
        "signal","action","qty","balance","position","equity","order_resp"
    ]

    balance = START_BALANCE
    position = 0.0
    last_order_ts = 0

    # ---------------------------------------------------------
    # WARM START
    # ---------------------------------------------------------
    logger.info("Warm-starting FeatureFactory V2 with %d candles…", WARMUP_KLINES)

    try:
        historic = await client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=WARMUP_KLINES)
        for k in historic:
            ohlcv.append((float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
        logger.info("Warm-start complete with %d candles", len(ohlcv))
    except Exception as e:
        logger.exception("Warm-start failed: %s", e)

    # After warm-start we can derive feature columns and ensure CSV header has them
    try:
        if len(ohlcv) >= 50:  # need some history to compute features
            df_live = pd.DataFrame(ohlcv, columns=["open","high","low","close","volume"])
            feats_df = build_features(df_live)
            # feature columns are those beyond the raw ohlcv
            feature_cols = [c for c in feats_df.columns if c not in ["open","high","low","close","volume","timestamp","close_time"]]
        else:
            feature_cols = []
    except Exception as e:
        logger.exception("Deriving feature columns failed: %s", e)
        feature_cols = []

    # ensure header exists and includes feature columns
    write_header_if_missing(LOG_CSV, base_cols, feature_cols)

    # ---------------------------------------------------------
    # LIVE LOOP
    # ---------------------------------------------------------
    async with ksocket as stream:
        while True:
            try:
                msg = await stream.recv()
                candle = msg["k"]
                # only handle closed candles
                if not candle["x"]:
                    await asyncio.sleep(0.01)
                    continue

                o = float(candle["o"])
                h = float(candle["h"])
                l = float(candle["l"])
                c = float(candle["c"])
                v = float(candle["v"])
                ts = dt.datetime.fromtimestamp(candle["T"] / 1000)

                # append candle to buffer
                ohlcv.append((o, h, l, c, v))

                # convert to DataFrame and build V2 features using same builder as training
                df_live = pd.DataFrame(ohlcv, columns=["open","high","low","close","volume"])

                try:
                    feats_df = build_features(df_live)   # returns DataFrame with all V2 features
                except Exception as e:
                    logger.exception("Feature builder exception: %s", e)
                    # wait for more data / avoid spamming
                    await asyncio.sleep(0.5)
                    continue

                if feats_df is None or feats_df.empty:
                    logger.info("Warmup… insufficient data for V2 features")
                    await asyncio.sleep(0.01)
                    continue

                # latest row = live features
                last_row = feats_df.iloc[-1].to_dict()

                # quick NaN guard
                if any(pd.isna(list(last_row.values()))):
                    logger.info("Warmup… features contain NaN, waiting for more candles")
                    await asyncio.sleep(0.01)
                    continue

                # Build X_live and align columns to model
                X_live = pd.DataFrame([last_row])

                # If model expects certain feature names, align / reindex (fill missing with 0)
                if hasattr(model, "feature_names_in_"):
                    model_cols = list(model.feature_names_in_)
                    # add any missing cols with zeros, and drop extras
                    for col in model_cols:
                        if col not in X_live.columns:
                            X_live[col] = 0.0
                    # keep only model_cols order
                    X_live = X_live[model_cols]
                else:
                    # best-effort: keep same columns as feats_df
                    pass

                # Prediction
                try:
                    pred = model.predict(X_live)
                    signal = int(pred[0])
                except Exception as e:
                    logger.exception("Prediction error: %s", e)
                    await asyncio.sleep(0.1)
                    continue

                # ---------------------------------------------------------
                # Execution logic (unchanged)
                # ---------------------------------------------------------
                action = "HOLD"
                qty = 0.0
                order_resp = None

                current_time = time.time()
                cooldown_ok = (current_time - last_order_ts >= ORDER_COOLDOWN)

                # BUY
                if signal == 1 and cooldown_ok and balance > 0:
                    qty = calc_order_qty(balance, RISK_PER_TRADE, c, lot_step)
                    if qty >= min_qty and qty > 0:
                        try:
                            order_resp = await client.create_test_order(
                                symbol=SYMBOL, side="BUY", type="MARKET", quantity=str(qty)
                            )
                            action = "BUY"
                            balance -= qty * c
                            position += qty
                            last_order_ts = current_time
                            logger.info("TEST BUY qty=%f price=%.2f", qty, c)
                        except Exception as e:
                            logger.exception("Test BUY failed: %s", e)

                # SELL (close full position)
                elif signal == 0 and cooldown_ok and position > 0:
                    qty = quantize_qty(position, lot_step)
                    if qty >= min_qty and qty > 0:
                        try:
                            order_resp = await client.create_test_order(
                                symbol=SYMBOL, side="SELL", type="MARKET", quantity=str(qty)
                            )
                            action = "SELL"
                            balance += qty * c
                            position = 0.0
                            last_order_ts = current_time
                            logger.info("TEST SELL qty=%f price=%.2f", qty, c)
                        except Exception as e:
                            logger.exception("Test SELL failed: %s", e)

                equity = balance + position * c

                # ---------------------------------------------------------
                # LOGGING - create a flat row merging ohlc + features + execution
                # ---------------------------------------------------------
                row = {
                    "timestamp": ts,
                    "open": o, "high": h, "low": l, "close": c, "volume": v,
                    **last_row,
                    "signal": signal,
                    "action": action,
                    "qty": qty,
                    "balance": round(balance, 8),
                    "position": round(position, 8),
                    "equity": round(equity, 8),
                    "order_resp": json.dumps(order_resp) if order_resp else ""
                }

                # append to CSV (will create columns in order of header created earlier)
                pd.DataFrame([row]).to_csv(LOG_CSV, mode="a", header=False, index=False)

                logger.info("[%s] Price %.2f | Sig=%d | Act=%s | Eq=%.2f",
                            ts.strftime("%H:%M:%S"), c, signal, action, equity)

            except Exception as e:
                logger.exception("Live loop error: %s", e)
                await asyncio.sleep(2)

    await client.close_connection()


if __name__ == "__main__":
    asyncio.run(main())


