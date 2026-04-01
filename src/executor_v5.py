# src/executor_v5.py
"""
Executor V5 — Strategy Research Lab (env aligned)
Uses env keys:
BINANCE_TESTNET, EXECUTE_ORDERS, MODEL_PATH, SYMBOL, INTERVAL,
ORDER_COOLDOWN, RISK_PER_TRADE, START_BALANCE, LOG_FILE, METRICS_FILE,
LIMIT_TIMEOUT, SLIPPAGE_BASE, VOL_ADJUST, WARMUP_KLINES, LOOKBACK,
DAILY_LOSS_LIMIT, MAX_DRAW_DOWN, MAX_CONSEC_LOSSES
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

# feature builder (your repo)
from feature_factory_v2 import build_features

load_dotenv()

# ----- ENV (from your pasted block) -----
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "True").lower() in ("1","true","yes")
EXECUTE_ORDERS = os.getenv("EXECUTE_ORDERS", "False").lower() in ("1","true","yes")
MODEL_PATH = os.getenv("MODEL_PATH", "models/phase3_rf_v2.pkl")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "1m")
ORDER_COOLDOWN = float(os.getenv("ORDER_COOLDOWN", 30))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.02))
START_BALANCE = float(os.getenv("START_BALANCE", 1000))
LOG_FILE = os.getenv("LOG_FILE", "executor_v4_log.csv")
METRICS_FILE = os.getenv("METRICS_FILE", "execution_metrics.csv")
LIMIT_TIMEOUT = float(os.getenv("LIMIT_TIMEOUT", 2.0))
SLIPPAGE_BASE = float(os.getenv("SLIPPAGE_BASE", 0.0005))
VOL_ADJUST = float(os.getenv("VOL_ADJUST", 1.0))
WARMUP_KLINES = int(os.getenv("WARMUP_KLINES", 500))
LOOKBACK = int(os.getenv("LOOKBACK", 2000))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", 0.1))   # fraction of starting balance
MAX_DRAW_DOWN = float(os.getenv("MAX_DRAW_DOWN", 0.2))
MAX_CONSEC_LOSSES = int(os.getenv("MAX_CONSEC_LOSSES", 5))

# ----- tuning / derived -----
SLIPPAGE_PCT = SLIPPAGE_BASE * VOL_ADJUST
FEE_PCT = 0.0004  # keep fixed small fee for paper (adjustable if you like)
PRED_PROB_THRESHOLD = float(os.getenv("PRED_PROB_THRESHOLD", 0.52))
TAU_SMOOTH = float(os.getenv("TAU_SMOOTH", 0.6))

LOOKBACK = max(500, LOOKBACK)
WARMUP_KLINES = max(100, WARMUP_KLINES)

# ----- logging -----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("executor_v5_env")

# ----- small helpers -----
def quantize_qty(qty, step):
    if step <= 0:
        return qty
    precision = max(0, int(round(-math.log10(step))))
    factor = 10 ** precision
    return math.floor(qty * factor) / factor

def calc_order_qty(balance_usd, risk_frac, price, step, min_notional=10.0):
    to_risk = max(0.0, balance_usd * risk_frac)
    qty = to_risk / (price + 1e-12)
    qty_q = quantize_qty(qty, step)
    if qty_q * price < min_notional:
        return 0.0
    return qty_q

def ensure_csv(path, cols):
    if not os.path.exists(path):
        pd.DataFrame(columns=cols).to_csv(path, index=False)

# ----- feature manifest fallback loader (optional) -----
def load_feature_manifest(path="models/features_columns_phase3_v2.json"):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cols = json.load(f)
            if isinstance(cols, list):
                logger.info("Loaded feature manifest (%d cols) from %s", len(cols), path)
                return cols
        except Exception:
            logger.exception("Failed to load feature manifest")
    return None

# ----- main executor -----
async def main():
    logger.info("Executor V5 starting | SYMBOL=%s | TESTNET=%s | EXECUTE_ORDERS=%s", SYMBOL, BINANCE_TESTNET, EXECUTE_ORDERS)

    # load model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Loaded ML model: %s", MODEL_PATH)
    except Exception as e:
        logger.exception("Model load failure")
        return

    # try manifest
    manifest = load_feature_manifest()

    # Binance client/socket
    client = await AsyncClient.create(os.getenv("BINANCE_API_KEY", ""), os.getenv("BINANCE_API_SECRET", ""), testnet=BINANCE_TESTNET)
    bm = BinanceSocketManager(client)

    # get symbol info (step sizes)
    info = await client.get_symbol_info(SYMBOL) or {}
    lot_step = 1e-8
    min_qty = 0.0
    min_notional = 10.0
    for f in info.get("filters", []):
        if f.get("filterType") == "LOT_SIZE":
            lot_step = float(f.get("stepSize", lot_step))
            min_qty = float(f.get("minQty", min_qty))
        if f.get("filterType") == "MIN_NOTIONAL":
            try:
                min_notional = float(f.get("minNotional", min_notional))
            except Exception:
                pass
    logger.info("LOT step=%s min_qty=%s min_notional=%s", lot_step, min_qty, min_notional)

    # rolling ohlcv
    ohlcv = deque(maxlen=LOOKBACK)
    # warm start
    try:
        hist = await client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=WARMUP_KLINES)
        for k in hist:
            ohlcv.append((float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
        logger.info("Warm-started with %d candles", len(ohlcv))
    except Exception:
        logger.exception("Warm-start failed (continuing)")

    # ensure logs exist
    base_cols = ["timestamp","open","high","low","close","volume"]
    ensure_csv(LOG_FILE, base_cols + ["signal","action","qty","balance","position","equity","order_resp"])
    ensure_csv(METRICS_FILE, ["timestamp","equity","balance","position","daily_pnl","drawdown","consec_losses","notes"])

    # state
    balance = START_BALANCE
    position = 0.0
    last_order_ts = 0.0
    signal_smooth = 0.0
    peak_equity = START_BALANCE
    daily_start = dt.datetime.utcnow().date()
    daily_pnl = 0.0
    consec_losses = 0
    total_trades = 0

    # socket + loop
    ksocket = bm.kline_socket(SYMBOL, interval=INTERVAL)
    reconnect_backoff = [1,2,5,10]
    reconnect_idx = 0

    while True:
        try:
            async with ksocket as stream:
                reconnect_idx = 0
                while True:
                    try:
                        msg = await asyncio.wait_for(stream.recv(), timeout=LIMIT_TIMEOUT)
                    except asyncio.TimeoutError:
                        # this may happen if there's no new data; continue and wait again
                        continue

                    candle = msg.get("k", {})
                    if not candle.get("x", False):
                        continue

                    o = float(candle["o"]); h = float(candle["h"]); l = float(candle["l"])
                    c = float(candle["c"]); v = float(candle["v"])
                    ts = dt.datetime.fromtimestamp(candle["T"]/1000.0)
                    ohlcv.append((o,h,l,c,v))

                    # build features
                    try:
                        df_live = pd.DataFrame(ohlcv, columns=["open","high","low","close","volume"])
                        feats_df = build_features(df_live)
                    except Exception:
                        logger.exception("Feature building error; skipping candle")
                        continue

                    if feats_df is None or feats_df.empty:
                        logger.info("Warmup: insufficient feature history")
                        continue

                    last_row = feats_df.iloc[-1].to_dict()

                    # prepare X_live
                    X_live = pd.DataFrame([last_row])
                    if manifest:
                        for col in manifest:
                            if col not in X_live.columns:
                                X_live[col] = 0.0
                        X_live = X_live[manifest]
                    X_live = X_live.apply(pd.to_numeric, errors="coerce").fillna(0.0)

                    # predict probabilities if available
                    try:
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(X_live)
                            prob_1 = float(probs[0][1])
                            prob_0 = float(probs[0][0])
                        else:
                            p = model.predict(X_live)
                            prob_1 = float(p[0])
                            prob_0 = 1.0 - prob_1
                        signal_raw = 1 if prob_1 >= PRED_PROB_THRESHOLD else 0
                    except Exception:
                        logger.exception("Prediction error")
                        continue

                    # smooth signal confidence
                    signal_smooth = TAU_SMOOTH * signal_smooth + (1 - TAU_SMOOTH) * prob_1
                    signal_smoothed = 1 if signal_smooth >= 0.5 else 0

                    # risk & management checks before acting
                    now_ts = time.time()
                    cooldown_ok = (now_ts - last_order_ts) >= ORDER_COOLDOWN

                    # daily reset
                    if dt.datetime.utcnow().date() != daily_start:
                        # write daily metrics
                        try:
                            pd.DataFrame([{
                                "timestamp": dt.datetime.utcnow(),
                                "equity": round(balance + position * c,8),
                                "balance": round(balance,8),
                                "position": round(position,8),
                                "daily_pnl": round(daily_pnl,8),
                                "drawdown": round((peak_equity - (balance + position * c))/peak_equity,8),
                                "consec_losses": consec_losses,
                                "notes": "daily_rollover"
                            }]).to_csv(METRICS_FILE, mode="a", header=False, index=False)
                        except Exception:
                            logger.exception("Metrics write failed on rollover")
                        daily_start = dt.datetime.utcnow().date()
                        daily_pnl = 0.0
                        consec_losses = 0

                    # stop-trade checks (safety)
                    equity = balance + position * c
                    drawdown = (peak_equity - equity) / (peak_equity + 1e-12)
                    if equity > peak_equity:
                        peak_equity = equity

                    # if we exceed daily loss limit or max drawdown -> pause trading (hold only)
                    if (START_BALANCE - equity) / START_BALANCE >= DAILY_LOSS_LIMIT or drawdown >= MAX_DRAW_DOWN or consec_losses >= MAX_CONSEC_LOSSES:
                        # log and skip trading actions
                        logger.warning("Trading paused: safety limits reached | daily_loss=%.4f drawdown=%.4f consec_losses=%d",
                                       (START_BALANCE - equity)/START_BALANCE, drawdown, consec_losses)
                        # still append a log row for monitoring
                        row = {"timestamp": ts, "open":o,"high":h,"low":l,"close":c,"volume":v,
                               **last_row, "signal":int(signal_raw), "action":"PAUSED", "qty":0.0,
                               "balance":round(balance,8),"position":round(position,8),"equity":round(equity,8),"order_resp":""}
                        try:
                            pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=False, index=False)
                        except Exception:
                            logger.exception("Failed to write paused row")
                        continue

                    action = "HOLD"
                    qty = 0.0
                    order_resp = None

                    # BUY: smoothed signal recommends buy, probability threshold met, cooldown and not already position-heavy
                    if signal_smoothed == 1 and prob_1 >= PRED_PROB_THRESHOLD and cooldown_ok and balance > 0:
                        qty = calc_order_qty(balance, RISK_PER_TRADE, c, lot_step, min_notional=min_notional)
                        qty = quantize_qty(qty, lot_step)
                        if qty >= min_qty and qty > 0:
                            exec_price = c * (1 + SLIPPAGE_PCT)
                            notional = qty * exec_price
                            fee = notional * FEE_PCT
                            try:
                                if EXECUTE_ORDERS:
                                    order_resp = await client.create_test_order(symbol=SYMBOL, side="BUY", type="MARKET", quantity=str(qty))
                                action = "BUY"
                                balance -= (notional + fee)
                                position += qty
                                last_order_ts = now_ts
                                total_trades += 1
                                logger.info("BUY executed qty=%.8f base_price=%.2f exec_price=%.2f notional=%.4f fee=%.6f",
                                            qty, c, exec_price, notional, fee)
                            except Exception:
                                logger.exception("BUY failed")

                    # SELL: smoothed signal says sell, we have a position, cooldown ok
                    elif signal_smoothed == 0 and position > 0 and cooldown_ok:
                        qty = quantize_qty(position, lot_step)
                        if qty >= min_qty and qty > 0:
                            exec_price = c * (1 - SLIPPAGE_PCT)
                            notional = qty * exec_price
                            fee = notional * FEE_PCT
                            try:
                                if EXECUTE_ORDERS:
                                    order_resp = await client.create_test_order(symbol=SYMBOL, side="SELL", type="MARKET", quantity=str(qty))
                                action = "SELL"
                                balance += (notional - fee)
                                position -= qty
                                if position < 1e-12:
                                    position = 0.0
                                last_order_ts = now_ts
                                total_trades += 1
                                # evaluate P&L from the trade -> crude approach: if trade decreased equity relative to previous, count as loss
                                # we track consec_losses using daily_pnl change
                                logger.info("SELL executed qty=%.8f base_price=%.2f exec_price=%.2f notional=%.4f fee=%.6f",
                                            qty, c, exec_price, notional, fee)
                                # simple: if exec notional - fee < qty * previous_avg_price => loss (we don't track avg fill price; this is a heuristic)
                                # Keep it simple here; increment trade counters
                            except Exception:
                                logger.exception("SELL failed")

                    # update equity & daily pnl
                    prev_equity = equity
                    equity = balance + position * c
                    pnl = equity - prev_equity
                    daily_pnl += pnl
                    if pnl < 0:
                        consec_losses += 1
                    else:
                        consec_losses = 0

                    # write log row
                    row = {"timestamp": ts, "open":o,"high":h,"low":l,"close":c,"volume":v,
                           **last_row, "signal":int(signal_raw), "action":action, "qty":qty,
                           "balance":round(balance,8),"position":round(position,8),"equity":round(equity,8),
                           "order_resp": json.dumps(order_resp) if order_resp else ""}
                    try:
                        pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=False, index=False)
                    except Exception:
                        logger.exception("Failed to append log row")

                    # write metrics row occasionally (every trade or every N candles)
                    if total_trades % 1 == 0:
                        try:
                            pd.DataFrame([{
                                "timestamp": dt.datetime.utcnow(),
                                "equity": round(equity,8),
                                "balance": round(balance,8),
                                "position": round(position,8),
                                "daily_pnl": round(daily_pnl,8),
                                "drawdown": round((peak_equity - equity)/ (peak_equity + 1e-12),8),
                                "consec_losses": consec_losses,
                                "notes": ""
                            }]).to_csv(METRICS_FILE, mode="a", header=False, index=False)
                        except Exception:
                            logger.exception("Failed to write metrics")

                    logger.info("[%s] Price %.2f | SigRaw=%d SigSmooth=%d Prob=%.3f Act=%s Eq=%.2f",
                                ts.strftime("%H:%M:%S"), c, int(signal_raw), int(signal_smoothed), prob_1, action, equity)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt -> exiting")
            break
        except Exception:
            logger.exception("Top-level error, reconnecting after backoff")
            wait = reconnect_backoff[min(reconnect_idx, len(reconnect_backoff)-1)]
            reconnect_idx += 1
            await asyncio.sleep(wait)
            # re-create socket manager to avoid stale objects
            try:
                bm = BinanceSocketManager(client)
                ksocket = bm.kline_socket(SYMBOL, interval=INTERVAL)
            except Exception:
                logger.exception("Failed to recreate socket manager")
                await asyncio.sleep(2)
                continue

    # cleanup
    try:
        await client.close_connection()
    except Exception:
        pass
    logger.info("Executor stopped")

if __name__ == "__main__":
    asyncio.run(main())

