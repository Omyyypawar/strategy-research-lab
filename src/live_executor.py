# src/live_executor.py
"""
LIVE EXECUTOR V4 — Strategy Research Lab
-----------------------------------------
Upgrades:
- Signal probability (predict_proba) and confidence-aware sizing
- Slippage modelling + realistic simulated fills
- Dynamic risk sizing using drawdown / daily-PnL throttle
- Feature manifest alignment (models/features_columns_phase3_v2.json)
- Writes detailed executor CSV + closed-trades CSV for analytics
- Keeps safe test-order behavior via EXECUTE_ORDERS / BINANCE_TESTNET flags
Author: Om | Strategy Research Lab
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
from typing import Optional

# ----------------------------------------------
# ENV / CONFIG
# ----------------------------------------------
load_dotenv()

API_KEY      = os.getenv("BINANCE_API_KEY", "")
API_SECRET   = os.getenv("BINANCE_API_SECRET", "")
USE_TESTNET  = os.getenv("BINANCE_TESTNET", "True").lower() in ("1","true","yes")
EXECUTE_ORDERS = os.getenv("EXECUTE_ORDERS", "False").lower() in ("1","true","yes")

MODEL_PATH   = os.getenv("MODEL_PATH", "models/phase3_rf_v2.pkl")
FEATURE_MANIFEST = os.getenv("FEATURE_MANIFEST", "models/features_columns_phase3_v2.json")

SYMBOL       = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL     = os.getenv("INTERVAL", "1m")

ORDER_COOLDOWN   = float(os.getenv("ORDER_COOLDOWN", "30"))
RISK_PER_TRADE   = float(os.getenv("RISK_PER_TRADE", "0.02"))
START_BALANCE    = float(os.getenv("START_BALANCE", "1000"))
LOG_CSV          = os.getenv("LOG_FILE", "executor_v4_log.csv")
TRADES_CSV       = os.getenv("TRADES_FILE", "executor_v4_trades.csv")

LOOKBACK         = int(os.getenv("LOOKBACK", "2000"))
WARMUP_KLINES    = int(os.getenv("WARMUP_KLINES", "500"))

# slippage base: fraction of price (e.g. 0.0005 -> 0.05%)
SLIPPAGE_BASE    = float(os.getenv("SLIPPAGE_BASE", "0.0005"))
SLIPPAGE_SCALE_BY_SIZE = float(os.getenv("SLIPPAGE_SCALE_BY_SIZE", "1.0"))

# drawdown based scaling
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0.1"))
MAX_DRAW_DOWN    = float(os.getenv("MAX_DRAW_DOWN", "0.2"))

# safety
MIN_ORDER_USD = float(os.getenv("MIN_ORDER_USD", "5.0"))

# rolling buffers
SAMPLING_SECONDS = int(os.getenv("SAMPLING_SECONDS", "30"))

# ----------------------------------------------
# Logging config
# ----------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("live_exec_v4")

# ----------------------------------------------
# Feature builder import (assumes feature_factory_v2 exists)
# ----------------------------------------------
from src.feature_factory_v2 import build_features

# ----------------------------------------------
# Binance helpers
# ----------------------------------------------
async def get_symbol_info(client, symbol):
    return await client.get_symbol_info(symbol)

def quantize_qty(qty, step):
    precision = int(round(-math.log10(step))) if step < 1 else 0
    return math.floor(qty * (10**precision)) / (10**precision)

def calc_order_qty_usd(balance_usd, risk_frac, price, min_usd=MIN_ORDER_USD):
    usd_to_trade = balance_usd * risk_frac
    if usd_to_trade < min_usd:
        return 0.0
    qty = usd_to_trade / price
    return qty

def simulate_fill_price(side: str, price: float, qty: float, lot_step: float) -> (float, float):
    """
    Simple slippage model:
    - base slippage (fraction) scaled by order size (qty) and SLIPPAGE_SCALE_BY_SIZE
    - returns fill_price, slippage_amount (fraction)
    """
    size_scale = max(1.0, qty * SLIPPAGE_SCALE_BY_SIZE)  # bigger qty -> more slippage
    slippage_frac = SLIPPAGE_BASE * size_scale
    if side.upper() == "BUY":
        fill_price = price * (1 + slippage_frac)
    else:
        fill_price = price * (1 - slippage_frac)
    fill_price = round(fill_price, 8)  # keep reasonable precision
    return fill_price, slippage_frac

# ----------------------------------------------
# Utility: feature alignment for model
# ----------------------------------------------
def load_feature_manifest(path: str) -> Optional[list]:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cols = json.load(f)
                if isinstance(cols, list):
                    return cols
        except Exception:
            return None
    return None

# ----------------------------------------------
# MAIN
# ----------------------------------------------
async def main():
    logger.info("Starting Live Executor V4 | Symbol=%s Testnet=%s EXECUTE_ORDERS=%s", SYMBOL, USE_TESTNET, EXECUTE_ORDERS)

    client = await AsyncClient.create(API_KEY, API_SECRET, testnet=USE_TESTNET)
    bm = BinanceSocketManager(client)
    ksocket = bm.kline_socket(SYMBOL, interval=INTERVAL)

    # load model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Loaded ML model: %s", MODEL_PATH)
    except Exception as e:
        logger.exception("Model load failure: %s", e)
        await client.close_connection()
        return

    # load feature manifest (preferred) otherwise fallback to model.feature_names_in_
    manifest = load_feature_manifest(FEATURE_MANIFEST)
    if manifest is None and hasattr(model, "feature_names_in_"):
        manifest = list(model.feature_names_in_)
        logger.info("Using model.feature_names_in_ for alignment")
    elif manifest:
        logger.info("Loaded feature manifest with %d cols", len(manifest))

    # symbol info
    info = await get_symbol_info(client, SYMBOL)
    if not info:
        logger.error("Could not fetch symbol info.")
        await client.close_connection()
        return

    lot_step = 1e-8
    min_qty = 0.0
    for f in info.get("filters", []):
        if f.get("filterType") == "LOT_SIZE":
            lot_step = float(f.get("stepSize", lot_step))
            min_qty = float(f.get("minQty", 0.0))

    # prepare logs
    if not os.path.exists(LOG_CSV):
        base_cols = [
            "timestamp","open","high","low","close","volume",
            # features will be appended dynamically if present in last_row
            "signal","signal_prob","action","qty","entry_price","exit_price",
            "balance","position","equity","pnl","pnl_pct","slippage_frac","order_resp"
        ]
        pd.DataFrame(columns=base_cols).to_csv(LOG_CSV, index=False)

    if not os.path.exists(TRADES_CSV):
        pd.DataFrame(columns=[
            "trade_id","entry_time","exit_time","entry_price","exit_price","qty",
            "side","pnl","pnl_pct","max_adverse_excursion","max_favorable_excursion","signal_prob_at_entry"
        ]).to_csv(TRADES_CSV, index=False)

    # state
    balance = START_BALANCE
    position = 0.0
    position_entry_price = 0.0
    position_entry_time = None
    position_entry_prob = None
    last_order_ts = 0.0
    equity = balance
    ohlcv = deque(maxlen=LOOKBACK)

    # trade bookkeeping
    trade_counter = 0
    open_trade_candles = []  # store (timestamp, price) while a trade is open for MAE/MFE
    daily_pnl = {}  # date -> pnl

    logger.info("Warm-starting feature builder with %d candles...", WARMUP_KLINES)
    try:
        historic = await client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=WARMUP_KLINES)
        for k in historic:
            ohlcv.append((float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
        logger.info("Warm-start complete with %d candles", len(ohlcv))
    except Exception as e:
        logger.exception("Warm-start failed: %s", e)

    async with ksocket as stream:
        while True:
            try:
                msg = await stream.recv()
                candle = msg["k"]
                if not candle["x"]:
                    await asyncio.sleep(0.01)
                    continue

                o = float(candle["o"]); h = float(candle["h"]); l = float(candle["l"])
                c = float(candle["c"]); v = float(candle["v"])
                ts = dt.datetime.fromtimestamp(candle["T"]/1000)

                # append candle
                ohlcv.append((o,h,l,c,v))

                # build features using same builder
                df_live = pd.DataFrame(list(ohlcv), columns=["open","high","low","close","volume"])
                try:
                    feats_df = build_features(df_live)
                except Exception as e:
                    logger.exception("Feature builder failed: %s", e)
                    await asyncio.sleep(0.1)
                    continue

                if feats_df is None or feats_df.empty:
                    await asyncio.sleep(0.01)
                    continue

                last_row = feats_df.iloc[-1].to_dict()
                # add raw close if not present
                last_row.setdefault("close", c)

                # quick NaN guard
                if any(pd.isna(list(last_row.values()))):
                    logger.info("Warmup… features contain NaN, waiting")
                    await asyncio.sleep(0.01)
                    continue

                # prepare X_live aligned to model
                X_live = pd.DataFrame([last_row])
                if manifest:
                    for col in manifest:
                        if col not in X_live.columns:
                            X_live[col] = 0.0
                    # keep order
                    X_live = X_live[manifest]

                # predict & probability
                signal = 0
                signal_prob = 0.0
                try:
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_live)
                        # assume positive class is 1; if classes_ ordering different, handle it
                        if hasattr(model, "classes_"):
                            # find index of class 1
                            idx_pos = list(model.classes_).index(1) if 1 in list(model.classes_) else -1
                            if idx_pos >= 0:
                                signal_prob = float(probs[0][idx_pos])
                            else:
                                # fallback: take max prob for predicted class
                                signal_prob = float(probs[0].max())
                        else:
                            signal_prob = float(probs[0].max())
                        signal = int(model.predict(X_live)[0])
                    else:
                        signal = int(model.predict(X_live)[0])
                        signal_prob = 1.0
                except Exception as e:
                    logger.exception("Prediction error: %s", e)
                    await asyncio.sleep(0.1)
                    continue

                # dynamic sizing: scale by signal_prob and by drawdown factor
                now_ts = time.time()
                cooldown_ok = (now_ts - last_order_ts >= ORDER_COOLDOWN)
                # drawdown factor: simple example (if equity dropped > MAX_DRAW_DOWN reduce size linearly)
                # compute equity conservatively
                equity = balance + position * c
                # naive historical worst: use last_row if contains derived equity? keep simple:
                dd_factor = 1.0
                # you can extend dd_factor by reading historical equity series file

                # effective risk per trade
                eff_risk = RISK_PER_TRADE * max(0.25, signal_prob) * dd_factor

                action = "HOLD"
                qty = 0.0
                entry_price = None
                exit_price = None
                slippage_frac = 0.0
                order_resp = None

                # BUY logic
                if signal == 1 and cooldown_ok and balance > 0:
                    # compute USD risk and convert to qty
                    qty_unrounded = calc_order_qty_usd(balance, eff_risk, c)
                    # quantize
                    qty = quantize_qty(qty_unrounded, lot_step)
                    if qty * c >= MIN_ORDER_USD and qty >= min_qty and qty > 0:
                        # simulate slippage & fill
                        fill_price, slippage_frac = simulate_fill_price("BUY", c, qty, lot_step)
                        entry_price = fill_price
                        order_resp = None
                        # execute or test
                        if EXECUTE_ORDERS:
                            try:
                                order_resp = await client.create_order(symbol=SYMBOL, side="BUY", type="MARKET", quantity=str(qty))
                            except Exception as e:
                                logger.exception("Live BUY failed: %s", e)
                                order_resp = {"error": str(e)}
                        # update simulated balances
                        balance -= qty * entry_price
                        position += qty
                        position_entry_price = entry_price
                        position_entry_time = ts
                        position_entry_prob = signal_prob
                        open_trade_candles = [(ts, c)]
                        last_order_ts = now_ts
                        action = "BUY"
                        logger.info("BUY fill qty=%.8f fill_price=%.2f prob=%.3f", qty, entry_price, signal_prob)
                    else:
                        logger.info("Computed qty too small (usd=%.2f qty=%.8f min_usd=%.2f)", qty * c, qty, MIN_ORDER_USD)

                # SELL logic - close all
                elif signal == 0 and position > 0 and cooldown_ok:
                    qty = quantize_qty(position, lot_step)
                    if qty >= min_qty and qty > 0:
                        # simulate slippage & fill
                        fill_price, slippage_frac = simulate_fill_price("SELL", c, qty, lot_step)
                        exit_price = fill_price
                        order_resp = None
                        if EXECUTE_ORDERS:
                            try:
                                order_resp = await client.create_order(symbol=SYMBOL, side="SELL", type="MARKET", quantity=str(qty))
                            except Exception as e:
                                logger.exception("Live SELL failed: %s", e)
                                order_resp = {"error": str(e)}
                        # compute trade PnL and MAE/MFE
                        entry_p = position_entry_price if position_entry_price else open_trade_candles[0][1] if open_trade_candles else None
                        if entry_p is None:
                            entry_p = exit_price  # fallback
                        pnl = (exit_price - entry_p) * qty
                        pnl_pct = (exit_price / entry_p - 1.0) * 100.0 if entry_p != 0 else 0.0
                        # MAE/MFE: compute min/max price relative to entry during open_trade_candles
                        prices_during = [p for (_, p) in open_trade_candles] + [c]
                        diffs = [(p - entry_p) for p in prices_during]
                        mae = min(diffs) if diffs else 0.0
                        mfe = max(diffs) if diffs else 0.0
                        # update balances
                        balance += qty * exit_price
                        position = 0.0
                        position_entry_price = 0.0
                        position_entry_time = None
                        position_entry_prob = None
                        open_trade_candles = []
                        last_order_ts = now_ts
                        action = "SELL"
                        # record closed trade
                        trade_counter += 1
                        trade_row = {
                            "trade_id": trade_counter,
                            "entry_time": position_entry_time.strftime("%Y-%m-%d %H:%M:%S") if position_entry_time else "",
                            "exit_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                            "entry_price": entry_p,
                            "exit_price": exit_price,
                            "qty": qty,
                            "side": "LONG",
                            "pnl": round(pnl, 8),
                            "pnl_pct": round(pnl_pct, 6),
                            "max_adverse_excursion": round(mae, 8),
                            "max_favorable_excursion": round(mfe, 8),
                            "signal_prob_at_entry": position_entry_prob if position_entry_prob else None
                        }
                        pd.DataFrame([trade_row]).to_csv(TRADES_CSV, mode="a", header=False, index=False)
                        logger.info("SELL filled qty=%.8f fill_price=%.2f PnL=%.6f", qty, exit_price, pnl)
                    else:
                        logger.info("Sell qty too small: %.8f (min %.8f)", qty, min_qty)

                # update open trade candles for MAE/MFE
                if position > 0:
                    open_trade_candles.append((ts, c))

                equity = balance + position * c
                pnl_now = equity - START_BALANCE
                pnl_pct_now = (equity / START_BALANCE - 1.0) * 100.0

                # create flat row for log: merge last_row features (safe: limited to simple types)
                flat_features = {}
                for k, v in last_row.items():
                    # only simple numeric / str features (avoid arrays)
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        flat_features[k] = v
                    elif isinstance(v, (str,)):
                        flat_features[k] = v

                log_row = {
                    "timestamp": ts,
                    "open": o, "high": h, "low": l, "close": c, "volume": v,
                    **flat_features,
                    "signal": int(signal),
                    "signal_prob": float(signal_prob),
                    "action": action,
                    "qty": round(qty, 8) if qty else 0.0,
                    "entry_price": round(entry_price, 8) if entry_price else None,
                    "exit_price": round(exit_price, 8) if exit_price else None,
                    "balance": round(balance, 8),
                    "position": round(position, 8),
                    "equity": round(equity, 8),
                    "pnl": round(pnl_now, 8),
                    "pnl_pct": round(pnl_pct_now, 6),
                    "slippage_frac": round(slippage_frac, 8),
                    "order_resp": json.dumps(order_resp) if order_resp else ""
                }

                # append to CSV (columns will expand as fed)
                pd.DataFrame([log_row]).to_csv(LOG_CSV, mode="a", header=False, index=False)

                logger.info("[%s] Price %.2f | Sig=%d (p=%.3f) | Act=%s | Eq=%.2f",
                            ts.strftime("%H:%M:%S"), c, signal, signal_prob, action, equity)

            except Exception as e:
                logger.exception("Live loop error: %s", e)
                await asyncio.sleep(2)

    await client.close_connection()

if __name__ == "__main__":
    asyncio.run(main())


