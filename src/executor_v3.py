"""
Executor V3 — Quant-Grade Execution Engine
Author: Om Pawar (Strategy Research Lab)
- Latency & slippage tracking
- Volatility-aware sizing
- Limit-maker -> market fallback routing
- Circuit breakers and risk controls
- Aligns live features to model.feature_names_in_
- Produces execution analytic logs
"""

import os
import asyncio
import json
import math
import time
import logging
import joblib
import pandas as pd
import numpy as np
import datetime as dt
from collections import deque, defaultdict
from dotenv import load_dotenv
from binance import AsyncClient, BinanceSocketManager

# ---------------------------
# CONFIG (from env)
# ---------------------------
load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
USE_TESTNET = os.getenv("BINANCE_TESTNET", "True").lower() in ("1", "true", "yes")
MODEL_PATH = os.getenv("MODEL_PATH", "models/phase3_rf_v2.pkl")

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "1m")

EXECUTE_ORDERS = os.getenv("EXECUTE_ORDERS", "False").lower() in ("1", "true", "yes")
ORDER_COOLDOWN = int(os.getenv("ORDER_COOLDOWN", "30"))           # seconds
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))      # fraction of equity
START_BALANCE = float(os.getenv("START_BALANCE", "1000"))
LOG_CSV = os.getenv("LOG_FILE", "executor_v3_log.csv")
METRICS_CSV = os.getenv("METRICS_FILE", "execution_metrics.csv")

LIMIT_TIMEOUT = float(os.getenv("LIMIT_TIMEOUT", "2.0"))         # seconds to wait for limit fill before market fallback
SLIPPAGE_BASE = float(os.getenv("SLIPPAGE_BASE", "0.0005"))     # 0.05% baseline slippage
VOL_ADJUST = float(os.getenv("VOL_ADJUST", "1.0"))              # multiplier for volatility-based slippage
WARMUP_KLINES = int(os.getenv("WARMUP_KLINES", "500"))
LOOKBACK = int(os.getenv("LOOKBACK", "2000"))

# Circuit breakers
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0.1"))  # 10% loss of starting equity
MAX_DRAW_DOWN = float(os.getenv("MAX_DRAW_DOWN", "0.2"))        # 20% max drawdown
MAX_CONSEC_LOSSES = int(os.getenv("MAX_CONSEC_LOSSES", "5"))

SAMPLING_SECONDS = int(os.getenv("SAMPLING_SECONDS", "30"))

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("executor_v3")

# ---------------------------
# Imports for your feature builder (must exist in project)
# Replace / adapt if you have an incremental engine instead.
# ---------------------------
try:
    from feature_factory_v2 import build_features
except Exception:
    # fallback to incremental_features if present
    try:
        from incremental_features import IncrementalFeatureEngine
        INC_ENGINE_AVAILABLE = True
    except Exception:
        INC_ENGINE_AVAILABLE = False
        build_features = None

# ---------------------------
# Helpers
# ---------------------------
def quantize_qty(qty, step):
    """Quantize to exchange lot step precision."""
    if step <= 0:
        return qty
    precision = int(round(-math.log10(step))) if step < 1 else 0
    return math.floor(qty * (10**precision)) / (10**precision)

def calc_size(balance, risk_pct, price, vol_regime=1.0):
    """
    Volatility-aware sizing:
    - reduce size when vol_regime > 1 (high volatility)
    - increase when vol_regime < 1 (low vol)
    """
    adj_risk = risk_pct / max(0.25, vol_regime)  # cap so not insane
    usd = balance * adj_risk
    qty = usd / price
    return qty

def align_features_for_model(X_row, model):
    """
    Ensure X_row (dict or DataFrame) has all model.feature_names_in_ columns
    missing -> filled with 0. maintain order
    """
    if hasattr(model, "feature_names_in_"):
        model_cols = list(model.feature_names_in_)
        X = pd.DataFrame([X_row])
        for c in model_cols:
            if c not in X.columns:
                X[c] = 0.0
        return X[model_cols]
    else:
        # best-effort return dataframe from dict
        return pd.DataFrame([X_row])

def now_ts():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

# ---------------------------
# Execution Engine
# ---------------------------
class ExecutorV3:
    def __init__(self, client, symbol, model):
        self.client = client
        self.symbol = symbol
        self.model = model

        # state
        self.balance = START_BALANCE
        self.position = 0.0
        self.last_order_ts = 0.0
        self.orders = []  # executed orders (dicts)
        self.trade_history = []  # per-trade analytics
        self.consec_losses = 0
        self.peak_equity = START_BALANCE

        # exchange rules
        self.lot_step = 1e-8
        self.min_qty = 0.0
        self.price_tick = 1e-8

        # rolling prices for vol/regime/slippage
        self.price_buf = deque(maxlen=200)
        self.vol_buf = deque(maxlen=200)

        # file initialisation
        if not os.path.exists(LOG_CSV):
            pd.DataFrame(columns=[
                "timestamp","symbol","open","high","low","close","volume",
                # feature columns are dynamic (we append full feature dict later)
                "signal","action","qty","price","balance","position","equity",
                "latency_ms","slippage_pct","order_type","order_resp"
            ]).to_csv(LOG_CSV, index=False)

        if not os.path.exists(METRICS_CSV):
            pd.DataFrame(columns=[
                "timestamp","equity","balance","position","PnL","peak_equity","drawdown","consec_losses"
            ]).to_csv(METRICS_CSV, index=False)

    async def load_symbol_info(self):
        info = await self.client.get_symbol_info(self.symbol)
        if not info:
            raise RuntimeError("Could not fetch symbol info from Binance.")
        for f in info.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                self.lot_step = float(f.get("stepSize", self.lot_step))
                self.min_qty = float(f.get("minQty", self.min_qty))
            if f.get("filterType") == "PRICE_FILTER":
                self.price_tick = float(f.get("tickSize", self.price_tick))

    def estimate_slippage(self, recent_volatility):
        """Adaptive slippage model: baseline + volatility multiplier"""
        return SLIPPAGE_BASE + VOL_ADJUST * max(0.0, recent_volatility)

    def update_price_buffers(self, price):
        self.price_buf.append(price)
        if len(self.price_buf) >= 2:
            rets = np.diff(np.array(self.price_buf)) / (np.array(self.price_buf)[:-1] + 1e-9)
            vol = np.std(rets[-50:]) if len(rets) >= 5 else np.std(rets)
            self.vol_buf.append(vol)

    def current_vol_regime(self):
        if len(self.vol_buf) == 0:
            return 1.0
        # ratio of recent vol to historical median
        recent = np.mean(list(self.vol_buf)[-10:]) if len(self.vol_buf) >= 10 else np.mean(self.vol_buf)
        med = np.median(list(self.vol_buf)) if len(self.vol_buf) >= 5 else recent + 1e-9
        return (recent / (med + 1e-9)) if med > 0 else 1.0

    def circuit_breaker_check(self, equity):
        # daily loss vs start balance
        daily_loss = max(0.0, START_BALANCE - equity) / START_BALANCE
        drawdown = max(0.0, self.peak_equity - equity) / max(1e-9, self.peak_equity)
        if daily_loss >= DAILY_LOSS_LIMIT:
            return False, f"Daily loss limit reached ({daily_loss*100:.2f}%)"
        if drawdown >= MAX_DRAW_DOWN:
            return False, f"Max drawdown triggered ({drawdown*100:.2f}%)"
        if self.consec_losses >= MAX_CONSEC_LOSSES:
            return False, "Max consecutive losses exceeded"
        return True, ""

    async def try_limit_then_market(self, side, quantity, price):
        """
        Try to place a limit order at price +/- small offset (to improve chance of maker)
        Wait LIMIT_TIMEOUT seconds: if not filled, cancel and fallback to market order.
        Returns: dict with order_resp, used_order_type, actual_price (fill), latency_ms
        """
        used_type = "NONE"
        order_resp = None
        latency_ms = None
        executed_price = None

        # build limit price: if BUY -> slightly below price to become maker; if SELL -> slightly above
        adj = float(self.price_tick) * 2
        limit_price = price - adj if side == "BUY" else price + adj
        limit_price = float(np.round(limit_price, int(round(-math.log10(self.price_tick))))) if self.price_tick < 1 else limit_price

        # Attempt LIMIT (make sure to not exceed precision)
        try:
            t0 = time.time()
            if EXECUTE_ORDERS:
                order_resp = await self.client.create_order(
                    symbol=self.symbol,
                    side=side,
                    type="LIMIT",
                    timeInForce="GTC",
                    quantity=str(quantity),
                    price=str(limit_price)
                )
            else:
                # test order simulation for testnet/local testing - use create_test_order if available
                if USE_TESTNET:
                    try:
                        order_resp = await self.client.create_test_order(
                            symbol=self.symbol,
                            side=side,
                            type="LIMIT",
                            timeInForce="GTC",
                            quantity=str(quantity),
                            price=str(limit_price)
                        )
                    except Exception:
                        order_resp = {"simulated":"limit_test_order", "price": limit_price, "qty": quantity}
                else:
                    order_resp = {"simulated":"limit_offline", "price": limit_price, "qty": quantity}

            # wait up to LIMIT_TIMEOUT for fill - NOTE: on test orders we cannot check fill so fallback logic below simulates
            await asyncio.sleep(LIMIT_TIMEOUT)

            # Check order status (only meaningful if we actually placed a real order)
            filled = False
            if EXECUTE_ORDERS:
                # fetch open orders and trade history for symbol - naive check
                try:
                    # try fetching order by id if present
                    if isinstance(order_resp, dict) and order_resp.get("orderId"):
                        status = await self.client.get_order(symbol=self.symbol, orderId=order_resp["orderId"])
                        if status.get("status") == "FILLED":
                            filled = True
                            executed_price = float(status.get("avgPrice") or status.get("price") or limit_price)
                    else:
                        # try scanning recent trades to infer fill - more expensive
                        trades = await self.client.get_my_trades(symbol=self.symbol, limit=5)
                        if trades:
                            # use last trade price as executed
                            executed_price = float(trades[-1]["price"])
                            filled = True
                except Exception:
                    filled = False
            else:
                # simulate a fill with some probability based on small spread - keep simple: we mark as not filled
                filled = False

            t1 = time.time()
            latency_ms = (t1 - t0) * 1000.0

            if filled:
                used_type = "LIMIT_FILLED"
                return {"order_resp": order_resp, "order_type": used_type, "price": executed_price, "latency_ms": latency_ms}
            else:
                # cancel order if real
                if EXECUTE_ORDERS and isinstance(order_resp, dict) and order_resp.get("orderId"):
                    try:
                        await self.client.cancel_order(symbol=self.symbol, orderId=order_resp["orderId"])
                    except Exception:
                        pass

                # fallback to market
                t0m = time.time()
                if EXECUTE_ORDERS:
                    mresp = await self.client.create_order(symbol=self.symbol, side=side, type="MARKET", quantity=str(quantity))
                    # fetch executed price from mresp or recent trades
                    price_taken = None
                    try:
                        if isinstance(mresp, dict) and mresp.get("fills"):
                            price_taken = float(mresp["fills"][-1]["price"])
                        else:
                            # best effort: fetch last trade
                            trades = await self.client.get_my_trades(symbol=self.symbol, limit=2)
                            if trades:
                                price_taken = float(trades[-1]["price"])
                    except Exception:
                        price_taken = price
                    t1m = time.time()
                    latency_ms = (t1m - t0m) * 1000.0
                    return {"order_resp": mresp, "order_type": "MARKET_FALLBACK", "price": price_taken or price, "latency_ms": latency_ms}
                else:
                    # simulate market fallback execution with small slippage penalty
                    t1m = time.time()
                    latency_ms = (t1m - t0) * 1000.0
                    # estimate price as market price (input price) +/- slippage
                    return {"order_resp": {"simulated": "market_fallback"}, "order_type": "MARKET_FALLBACK", "price": price * (1.0 + SLIPPAGE_BASE if side == "BUY" else 1.0 - SLIPPAGE_BASE), "latency_ms": latency_ms}
        except Exception as e:
            logger.exception("Order routing exception: %s", e)
            return {"order_resp": {"error": str(e)}, "order_type": "ERROR", "price": price, "latency_ms": 0.0}

    async def on_candle(self, df_ohlcv):
        """
        df_ohlcv: DataFrame of recent candles with columns ['open','high','low','close','volume', ...]
        Called for each closed candle.
        """
        # last candle
        last = df_ohlcv.iloc[-1]
        o, h, l, c, v = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"]), float(last["volume"])
        ts = dt.datetime.utcnow()

        # maintain price buffer and volatility
        self.update_price_buffers(c)
        vol_regime = self.current_vol_regime()
        recent_vol = np.mean(self.vol_buf) if len(self.vol_buf) > 0 else 0.0
        slippage_est = self.estimate_slippage(recent_vol)

        # Build features using your feature factory (must be same as training)
        try:
            feats_df = build_features(df_ohlcv) if build_features is not None else None
            if feats_df is None or feats_df.empty:
                logger.info("V3: insufficient data for feature builder.")
                return
            feats = feats_df.iloc[-1].to_dict()
        except Exception as e:
            logger.exception("Feature build error: %s", e)
            return

        # Merge price into features if model expects
        feats["close"] = c

        # Align features
        X_live = align_features_for_model(feats, self.model)

        # Predict
        try:
            pred = self.model.predict(X_live)
            signal = int(pred[0])
        except Exception as e:
            logger.exception("Prediction failure: %s", e)
            return

        # Anti-duplicate / cooldown logic
        now = time.time()
        if now - self.last_order_ts < ORDER_COOLDOWN:
            cooldown_ok = False
        else:
            cooldown_ok = True

        # Decide action
        action = "HOLD"
        qty = 0.0
        order_resp = None
        order_type = None
        latency_ms = None
        executed_price = None
        slippage_pct = 0.0

        equity = self.balance + self.position * c

        # Circuit breaker check
        ok, reason = self.circuit_breaker_check(equity)
        if not ok:
            logger.warning("Circuit breaker active: %s", reason)
            action = "PAUSED"
            self.log_row(ts, o, h, l, c, v, feats, signal, action, qty, None, latency_ms, slippage_pct, order_type, order_resp)
            return

        # Signal handling: BUY(1) -> open/increase, SELL(0) -> close
        if signal == 1 and cooldown_ok:
            # compute volatility-adjusted size
            raw_qty = calc_size(self.balance, RISK_PER_TRADE, c, vol_regime=vol_regime)
            qty = quantize_qty(raw_qty, self.lot_step)
            if qty >= self.min_qty and qty > 0:
                # place order via routing
                routing = await self.try_limit_then_market("BUY", qty, c)
                order_resp = routing.get("order_resp")
                order_type = routing.get("order_type")
                executed_price = routing.get("price", c)
                latency_ms = routing.get("latency_ms", None)
                slippage_pct = (executed_price - c) / (c + 1e-9)
                # update state (simple: assume filled for simulation; for real we must parse order_resp)
                self.balance -= qty * executed_price
                self.position += qty
                self.last_order_ts = time.time()
                action = "BUY"
        elif signal == 0 and self.position > 0 and cooldown_ok:
            # Sell full position
            qty = quantize_qty(self.position, self.lot_step)
            if qty >= self.min_qty and qty > 0:
                routing = await self.try_limit_then_market("SELL", qty, c)
                order_resp = routing.get("order_resp")
                order_type = routing.get("order_type")
                executed_price = routing.get("price", c)
                latency_ms = routing.get("latency_ms", None)
                slippage_pct = (c - executed_price) / (c + 1e-9) if order_type else 0.0
                self.balance += qty * executed_price
                # compute per-trade PnL to update consec losses/peak equity
                pnl = (executed_price - (0 if self.position == 0 else executed_price)) * qty  # simplified placeholder
                # set position to zero
                self.position = 0.0
                self.last_order_ts = time.time()
                action = "SELL"

        equity = self.balance + self.position * c
        # update peak equity & consecutive losses (naive: if equity declined from previous)
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.consec_losses = 0
        else:
            # if last trade was SELL/BUY that decreased equity, increment consec_losses
            # we infer loss if equity < previous peak (coarse)
            self.consec_losses += 0 if action == "HOLD" else 1

        # Log row
        self.log_row(ts, o, h, l, c, v, feats, signal, action, qty, executed_price, latency_ms, slippage_pct, order_type, order_resp)

        # Append metrics snapshot
        self.log_metrics_snapshot(equity)

    def log_row(self, ts, o, h, l, c, v, feats, signal, action, qty, price, latency_ms, slippage_pct, order_type, order_resp):
        # Flatten features - keep as separate columns by prefixing "f_"
        flat_feats = {f"f_{k}": (v if not isinstance(v, (list, dict)) else json.dumps(v)) for k, v in feats.items()}
        row = {
            "timestamp": ts.isoformat(),
            "symbol": self.symbol,
            "open": o, "high": h, "low": l, "close": c, "volume": v,
            "signal": signal,
            "action": action,
            "qty": qty,
            "price": price or c,
            "balance": round(self.balance, 8),
            "position": round(self.position, 8),
            "equity": round(self.balance + self.position * c, 8),
            "latency_ms": latency_ms or 0.0,
            "slippage_pct": slippage_pct or 0.0,
            "order_type": order_type or "",
            "order_resp": json.dumps(order_resp) if order_resp else ""
        }
        # merge flat_feats
        row.update(flat_feats)
        try:
            pd.DataFrame([row]).to_csv(LOG_CSV, mode="a", header=False, index=False)
        except Exception as e:
            logger.exception("Failed to append log row: %s", e)

    def log_metrics_snapshot(self, equity):
        drawdown = (self.peak_equity - equity) / max(1e-9, self.peak_equity)
        row = {
            "timestamp": now_ts(),
            "equity": round(equity, 6),
            "balance": round(self.balance, 6),
            "position": round(self.position, 6),
            "PnL": round(equity - START_BALANCE, 6),
            "peak_equity": round(self.peak_equity, 6),
            "drawdown": round(drawdown, 6),
            "consec_losses": int(self.consec_losses)
        }
        try:
            pd.DataFrame([row]).to_csv(METRICS_CSV, mode="a", header=False, index=False)
        except Exception as e:
            logger.exception("Failed to append metrics row: %s", e)

# ---------------------------
# Main async runner
# ---------------------------
async def main():
    logger.info("Starting Executor V3 | Symbol=%s Testnet=%s Execute=%s", SYMBOL, USE_TESTNET, EXECUTE_ORDERS)

    client = await AsyncClient.create(API_KEY, API_SECRET, testnet=USE_TESTNET)
    bm = BinanceSocketManager(client)
    ksocket = bm.kline_socket(SYMBOL, interval=INTERVAL)

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Loaded model: %s", MODEL_PATH)
    except Exception as e:
        logger.exception("Model load failed: %s", e)
        await client.close_connection()
        return

    exec_engine = ExecutorV3(client, SYMBOL, model)
    try:
        await exec_engine.load_symbol_info()
    except Exception as e:
        logger.exception("Failed to load symbol info: %s", e)
        await client.close_connection()
        return

    # Warm-start: fetch historical klines to feed feature builder
    try:
        historic = await client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=WARMUP_KLINES)
        ohlcv = deque(maxlen=LOOKBACK)
        import pandas as pd
        for k in historic:
            o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
            ohlcv.append((o, h, l, c, v))
            exec_engine.update_price_buffers(c)
        logger.info("Warm-start complete with %d candles", len(ohlcv))
    except Exception as e:
        logger.exception("Warm start failed: %s", e)
        ohlcv = deque(maxlen=LOOKBACK)

    # Stream handler
    async with ksocket as stream:
        while True:
            try:
                msg = await stream.recv()
                candle = msg["k"]
                if not candle["x"]:
                    await asyncio.sleep(0.01)
                    continue

                o = float(candle["o"]); h = float(candle["h"]); l = float(candle["l"]); c = float(candle["c"]); v = float(candle["v"])
                ohlcv.append((o, h, l, c, v))
                df_live = pd.DataFrame(list(ohlcv), columns=["open","high","low","close","volume"])

                # call engine logic
                await exec_engine.on_candle(df_live)

                await asyncio.sleep(0.01)
            except Exception as e:
                logger.exception("Stream loop error: %s", e)
                await asyncio.sleep(1)

    await client.close_connection()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Executor V3 stopped by user")
