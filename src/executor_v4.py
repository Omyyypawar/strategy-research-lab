# src/executor_v4.py
"""
Executor V4 — Institutional Execution Layer
-------------------------------------------
Features:
- Spread-aware execution using orderbook depth (estimate VWAP + slippage)
- Adaptive order sizing (cuts size when expected impact is high)
- Ensemble signal: ML model + regime filter + micro-vol consensus
- Simulated or real orders (controlled by EXECUTE_ORDERS env)
- Logging: extended CSV with slippage, spread, exec_type, orderbook snapshot
- Safe model column alignment (uses model.feature_names_in_ when available)

Author: Om | Strategy Research Lab
"""

import os
import asyncio
import joblib
import pandas as pd
import numpy as np
import datetime as dt
import json
import math
import time
import logging
from collections import deque
from dotenv import load_dotenv
from binance import AsyncClient, BinanceSocketManager

# -------------------------
# CONFIG / ENV
# -------------------------
load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
USE_TESTNET = os.getenv("BINANCE_TESTNET", "True").lower() in ("1","true","yes")

MODEL_PATH = os.getenv("MODEL_PATH", "models/phase3_rf_v2.pkl")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "1m")

EXECUTE_ORDERS = os.getenv("EXECUTE_ORDERS", "False").lower() in ("1","true","yes")
ORDER_COOLDOWN = int(os.getenv("ORDER_COOLDOWN", "30"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))
START_BALANCE = float(os.getenv("START_BALANCE", "1000"))
LOG_CSV = os.getenv("LOG_FILE", "executor_v4_log.csv")

LOOKBACK = int(os.getenv("LOOKBACK", "2000"))
WARMUP_KLINES = int(os.getenv("WARMUP_KLINES", "500"))
SAMPLING_SECONDS = int(os.getenv("SAMPLING_SECONDS", "30"))

# Execution parameters
MAX_SLIPPAGE_PCT = float(os.getenv("MAX_SLIPPAGE_PCT", "0.005"))   # 0.5% default acceptable
MAX_IMPACT_PCT = float(os.getenv("MAX_IMPACT_PCT", "0.01"))        # 1% impact threshold
POST_ONLY_THRESHOLD = float(os.getenv("POST_ONLY_THRESHOLD", "0.0005"))  # if spread > threshold use limit

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("executor_v4")

# -------------------------
# Helpers: quantize / qty
# -------------------------
def quantize(qty, step):
    """Round down to step precision."""
    if step >= 1:
        return math.floor(qty)
    precision = int(round(-math.log10(step)))
    return math.floor(qty * (10 ** precision)) / (10 ** precision)

def calc_order_qty(balance_usd, risk_frac, price, step):
    usd_to_trade = balance_usd * risk_frac
    qty = usd_to_trade / (price + 1e-12)
    return quantize(qty, step)

# -------------------------
# Orderbook / slippage utils
# -------------------------
def estimate_market_vwap(orderbook, side, qty):
    """
    Estimate VWAP to fill `qty` on given side using top-of-book levels.
    orderbook: dict with 'bids' and 'asks', each list [price, size]
    side: 'BUY' or 'SELL' — BUY consumes asks, SELL consumes bids
    returns: (vwap_price, est_slippage_pct, filled_qty)
    """
    levels = orderbook["asks"] if side == "BUY" else orderbook["bids"]
    remaining = qty
    total_value = 0.0
    filled = 0.0
    for p_str, s_str in levels:
        p = float(p_str); s = float(s_str)
        take = min(remaining, s)
        total_value += take * p
        filled += take
        remaining -= take
        if remaining <= 1e-12:
            break
    if filled <= 0:
        return None, None, 0.0
    vwap = total_value / filled
    # mid price for reference
    top_bid = float(orderbook["bids"][0][0]) if orderbook["bids"] else vwap
    top_ask = float(orderbook["asks"][0][0]) if orderbook["asks"] else vwap
    mid = 0.5 * (top_bid + top_ask)
    slippage = (vwap - mid) / (mid + 1e-12)
    if side == "SELL":
        slippage = -slippage  # for clarity: negative means receive worse price relative to mid
    return vwap, slippage, filled

async def fetch_orderbook(client, symbol, limit=100):
    ob = await client.get_order_book(symbol=symbol, limit=limit)
    # Binance returns bids/asks as lists of [price, qty]
    return ob

# -------------------------
# Feature alignment util
# -------------------------
def align_features_to_model(X_df, model):
    """Ensure X_df has the same columns as model.feature_names_in_ (fill missing with 0)."""
    if hasattr(model, "feature_names_in_"):
        model_cols = list(model.feature_names_in_)
        for c in model_cols:
            if c not in X_df.columns:
                X_df[c] = 0.0
        X_df = X_df[model_cols]
    return X_df

# -------------------------
# Ensemble scoring
# -------------------------
def ensemble_score(model, X_row, feats_row):
    """
    Compose an ensemble confidence score using:
    - model predict_proba (if available)
    - regime filter (vol_regime, vpin etc)
    - micro-vol indicator (avg_vol_20 vs recently)
    Returns: (binary_final_signal, confidence_score [0..1], components dict)
    """
    comp = {}
    # model probability
    try:
        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba(X_row)[0, 1])
        else:
            # fallback to predict -> 0/1 with moderate confidence
            p = float(model.predict(X_row)[0]) * 0.6 + 0.4 * 0.5
    except Exception:
        p = 0.5
    comp["model_prob"] = p

    # regime filter: prefer trades when vol_regime near 1 (not extreme)
    vol_regime = float(feats_row.get("vol_regime", 1.0))
    comp["regime_ok"] = 1.0 if 0.3 < vol_regime < 3.0 else 0.0

    # micro-vol filter (small recent volatility preferred)
    vol_short = float(feats_row.get("vol_short", 0.0))
    vol_long = float(feats_row.get("vol_long", 1e-9))
    micro_ok = 1.0 if vol_short < (vol_long * 1.2 + 1e-12) else 0.0
    comp["micro_ok"] = micro_ok

    # simple voting with weights
    conf = 0.6 * p + 0.2 * comp["regime_ok"] + 0.2 * comp["micro_ok"]
    final = 1 if conf > 0.55 else 0
    return final, float(conf), comp

# -------------------------
# CSV columns / prepare log
# -------------------------
LOG_COLUMNS = [
    "timestamp","open","high","low","close","volume",
    # core features will be merged dynamically (from feature builder)
    # execution columns
    "signal_ml","signal_ensemble","ensemble_conf","ensemble_components",
    "exec_type","order_qty","order_price_est","est_slippage_pct","spread_pct",
    "balance","position","equity","order_resp","orderbook_snapshot"
]

# -------------------------
# MAIN
# -------------------------
async def main():
    logger.info("Starting Executor V4 | Symbol=%s Testnet=%s", SYMBOL, USE_TESTNET)

    client = await AsyncClient.create(API_KEY, API_SECRET, testnet=USE_TESTNET)
    bm = BinanceSocketManager(client)
    ksocket = bm.kline_socket(SYMBOL, interval=INTERVAL)

    # load model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Loaded model: %s", MODEL_PATH)
    except Exception as e:
        logger.exception("Model load failure: %s", e)
        await client.close_connection()
        return

    # fetch symbol info for lot step
    info = await client.get_symbol_info(SYMBOL)
    if not info:
        logger.error("Could not get symbol info from Binance")
        await client.close_connection()
        return

    lot_step = 1e-8
    min_qty = 0.0
    for f in info.get("filters", []):
        if f.get("filterType") == "LOT_SIZE":
            lot_step = float(f.get("stepSize", lot_step))
            min_qty = float(f.get("minQty", 0.0))

    # Rolling buffer
    ohlcv = deque(maxlen=LOOKBACK)

    # Prepare CSV
    if not os.path.exists(LOG_CSV):
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_CSV, index=False)

    balance = float(START_BALANCE)
    position = 0.0
    last_order_ts = 0.0

    # We assume you have a feature builder that produces same features as training.
    # Import locally to avoid circular import issues (user should have feature_factory_v2)
    try:
        from src.feature_factory_v2 import build_features
    except Exception as e:
        logger.exception("feature builder import failed: %s", e)
        await client.close_connection()
        return

    # Warm-start
    logger.info("Warm-starting with %d klines...", WARMUP_KLINES)
    try:
        historic = await client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=WARMUP_KLINES)
        for k in historic:
            ohlcv.append((float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
        logger.info("Warm-start complete with %d candles", len(ohlcv))
    except Exception as e:
        logger.exception("Warm-start failed: %s", e)

    # Live loop
    async with ksocket as stream:
        while True:
            try:
                msg = await stream.recv()
                candle = msg["k"]
                if not candle["x"]:
                    await asyncio.sleep(0.01); continue

                o = float(candle["o"]); h = float(candle["h"]); l = float(candle["l"])
                c = float(candle["c"]); v = float(candle["v"])
                ts = dt.datetime.fromtimestamp(candle["T"] / 1000)

                ohlcv.append((o,h,l,c,v))

                # build features using exact builder used for training
                df_live = pd.DataFrame(ohlcv, columns=["open","high","low","close","volume"])
                feats_df = build_features(df_live)
                if feats_df is None or feats_df.empty:
                    logger.info("Warmup: insufficient features")
                    await asyncio.sleep(0.01); continue

                last_feats = feats_df.iloc[-1].to_dict()
                # Add essential market columns if missing
                last_feats.setdefault("close", c)
                last_feats.setdefault("vol_short", last_feats.get("vol_short", 0.0))
                last_feats.setdefault("vol_long", last_feats.get("vol_long", 0.0))
                last_feats.setdefault("vol_regime", last_feats.get("vol_regime", 1.0))

                # Build X_live aligned to model
                X_live = pd.DataFrame([last_feats])
                X_live = align_features_to_model(X_live, model)

                # ML signal and ensemble
                try:
                    ml_pred = int(model.predict(X_live)[0])
                except Exception as e:
                    logger.exception("ML pred error: %s", e)
                    await asyncio.sleep(0.05); continue

                final_sig, conf, comps = ensemble_score(model, X_live, last_feats)

                # Fetch orderbook snapshot (top 100)
                orderbook = await fetch_orderbook(client, SYMBOL, limit=100)
                top_bid = float(orderbook["bids"][0][0]) if orderbook["bids"] else c
                top_ask = float(orderbook["asks"][0][0]) if orderbook["asks"] else c
                spread_pct = (top_ask - top_bid) / (0.5*(top_ask+top_bid) + 1e-12)

                # Decide qty based on dynamic sizing: reduce size if estimated impact large
                base_qty = calc_order_qty(balance, RISK_PER_TRADE, c, lot_step)

                # Estimate market vwap & slippage for intended side
                side = "BUY" if final_sig == 1 else "SELL"
                est_vwap, est_slippage, filled = estimate_market_vwap(orderbook, side, base_qty)
                est_slippage_pct = float(est_slippage) if est_slippage is not None else 0.0

                # Adaptive sizing: if est_slippage abs > MAX_SLIPPAGE_PCT reduce qty proportionally
                qty = base_qty
                exec_type = "SIM_MARKET"
                if est_slippage_pct is not None and abs(est_slippage_pct) > MAX_SLIPPAGE_PCT:
                    factor = MAX_SLIPPAGE_PCT / (abs(est_slippage_pct) + 1e-12)
                    qty = quantize(base_qty * factor, lot_step)
                    exec_type = "SIM_MARKET_REDUCED"

                # If spread is large, prefer limit/post-only (simulated)
                if spread_pct > POST_ONLY_THRESHOLD:
                    exec_type = "SIM_LIMIT_POSTONLY"
                    # For post-only simulate we assume we get mid-price if not removed — conservative: mid +/- tiny
                    est_price = 0.5*(top_bid + top_ask)
                else:
                    est_price = est_vwap if est_vwap is not None else c

                # final safety guards (min qty)
                if qty < min_qty or qty <= 0:
                    # no trade
                    action = "HOLD"
                    order_resp = None
                    logger.info("[%s] Price %.2f | Sig(final)=%d | Act=HOLD (qty too small) | Eq=%.2f", ts.strftime("%H:%M:%S"), c, final_sig, balance+position*c)
                else:
                    # Cooldown
                    now_ts = time.time()
                    if now_ts - last_order_ts < ORDER_COOLDOWN:
                        action = "HOLD"
                        order_resp = None
                        logger.info("[%s] Price %.2f | Sig(final)=%d | Act=HOLD (cooldown) | Eq=%.2f", ts.strftime("%H:%M:%S"), c, final_sig, balance+position*c)
                    else:
                        # Execution either simulated or real (but still safe)
                        if EXECUTE_ORDERS:
                            # For safety prefer test endpoints if available, else create real with proper params
                            try:
                                # Use market test order when possible
                                order_resp = await client.create_test_order(
                                    symbol=SYMBOL,
                                    side=side,
                                    type="MARKET",
                                    quantity=str(qty)
                                )
                                # simulate fill bookkeeping
                                if side == "BUY":
                                    position += qty
                                    balance -= qty * est_price
                                else:
                                    position = max(0.0, position - qty)
                                    balance += qty * est_price
                                action = side
                                last_order_ts = now_ts
                                logger.info("[%s] EXECUTED %s qty=%.8f price_est=%.2f conf=%.3f", ts.strftime("%H:%M:%S"), side, qty, est_price, conf)
                            except Exception as e:
                                order_resp = {"error": str(e)}
                                action = "ERROR"
                                logger.exception("Order execution failed: %s", e)
                        else:
                            # Simulated execution: record estimation as "order_resp"
                            order_resp = {"simulated": True, "side": side, "qty": qty, "est_price": est_price, "est_slippage_pct": est_slippage_pct}
                            if side == "BUY":
                                position += qty
                                balance -= qty * est_price
                            else:
                                position = max(0.0, position - qty)
                                balance += qty * est_price
                            action = side
                            last_order_ts = now_ts
                            logger.info("[%s] SIM %s qty=%.8f price_est=%.2f conf=%.3f exec_type=%s", ts.strftime("%H:%M:%S"), side, qty, est_price, conf, exec_type)

                equity = balance + position * c

                # Build log row
                row = {
                    "timestamp": ts,
                    "open": o, "high": h, "low": l, "close": c, "volume": v,
                    "signal_ml": int(ml_pred),
                    "signal_ensemble": int(final_sig),
                    "ensemble_conf": float(conf),
                    "ensemble_components": json.dumps(comps),
                    "exec_type": exec_type,
                    "order_qty": float(qty),
                    "order_price_est": float(est_price),
                    "est_slippage_pct": float(est_slippage_pct) if est_slippage_pct is not None else 0.0,
                    "spread_pct": float(spread_pct),
                    "balance": round(balance, 8),
                    "position": round(position, 8),
                    "equity": round(equity, 8),
                    "order_resp": json.dumps(order_resp) if order_resp is not None else "",
                    "orderbook_snapshot": json.dumps({
                        "top_bid": orderbook["bids"][:5],
                        "top_ask": orderbook["asks"][:5]
                    })
                }

                # merge core feature columns (flat)
                for k, v in last_feats.items():
                    # avoid overriding execution fields
                    if k in row:
                        continue
                    # keep primitive types
                    try:
                        row[k] = float(v) if (isinstance(v, (int,float,np.number)) or str(v).replace('.','',1).isdigit()) else v
                    except Exception:
                        row[k] = v

                # append to CSV
                pd.DataFrame([row]).to_csv(LOG_CSV, mode="a", header=False, index=False)

            except Exception as e:
                logger.exception("Live loop error: %s", e)
                await asyncio.sleep(1)

    await client.close_connection()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Executor V4 terminated by user.")
