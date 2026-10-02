"""
Incremental Feature Engine for live 30s ticks
Keeps rolling state and returns features for the latest tick only.
Author: Om | Strategy Research Lab
"""

import collections
import numpy as np
import pandas as pd


class RollingState:
    def __init__(self, maxlen=2000):
        self.opens = collections.deque(maxlen=maxlen)
        self.highs = collections.deque(maxlen=maxlen)
        self.lows = collections.deque(maxlen=maxlen)
        self.closes = collections.deque(maxlen=maxlen)
        self.vols = collections.deque(maxlen=maxlen)
        self.tr_list = collections.deque(maxlen=maxlen)

    def append(self, o, h, l, c, v):
        self.opens.append(float(o))
        self.highs.append(float(h))
        self.lows.append(float(l))
        self.closes.append(float(c))
        self.vols.append(float(v))

    def as_series(self):
        return {
            "open": pd.Series(self.opens),
            "high": pd.Series(self.highs),
            "low": pd.Series(self.lows),
            "close": pd.Series(self.closes),
            "volume": pd.Series(self.vols),
        }


class IncrementalFeatureEngine:
    def __init__(self, sampling_seconds=30, atr_window=14, z_window=20, vol_window=20):
        self.state = RollingState(maxlen=2000)
        self.sampling_seconds = sampling_seconds
        self.atr_window = atr_window
        self.z_window = z_window
        self.vol_window = vol_window

        # EMA incremental state
        self.ema_fast = None
        self.ema_slow = None
        self.ema_fast_span = 8
        self.ema_slow_span = 21

    # ------------------------------
    # EMA alpha values
    # ------------------------------
    def _alpha(self, span):
        return 2.0 / (span + 1.0)

    def _update_ema(self, price):
        price = float(price)
        if self.ema_fast is None:
            self.ema_fast = price
            self.ema_slow = price
        else:
            af = self._alpha(self.ema_fast_span)
            aslow = self._alpha(self.ema_slow_span)
            self.ema_fast = af * price + (1 - af) * self.ema_fast
            self.ema_slow = aslow * price + (1 - aslow) * self.ema_slow

    # ------------------------------
    # True Range for ATR
    # ------------------------------
    def _update_tr(self, high, low, prev_close):
        if prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        self.state.tr_list.append(tr)
        return tr

    # ------------------------------
    # MAIN METHOD — append tick
    # ------------------------------
    def append_tick(self, o, h, l, c, v):
        """Append tick and compute latest features (lag-safe)."""
        self.state.append(o, h, l, c, v)

        s = self.state.as_series()
        close_series = s["close"]
        n = len(close_series)

        # Prediction horizon = 5 minutes = 10 samples
        steps = int(5 * 60 / self.sampling_seconds)

        # Safe backward returns
        def pct_change_steps(series, steps):
            if len(series) <= steps:
                return np.nan
            prev = series.iloc[-1 - steps]
            if prev == 0:
                return np.nan
            return (series.iloc[-1] / prev) - 1.0

        ret_1m = pct_change_steps(close_series, int(60 / self.sampling_seconds))
        ret_5m = pct_change_steps(close_series, steps)
        ret_15m = pct_change_steps(close_series, int(15 * 60 / self.sampling_seconds))

        # EMA update
        self._update_ema(close_series.iloc[-1])
        ema_fast = self.ema_fast
        ema_slow = self.ema_slow
        ema_trend = ema_fast - ema_slow if ema_fast is not None else np.nan
        ema_trend_norm = ema_trend / (close_series.iloc[-1] if close_series.iloc[-1] != 0 else np.nan)

        # Rolling std
        if n >= 5:
            rolling_std_5 = close_series.iloc[-5:].std()
        else:
            rolling_std_5 = close_series.std() if n > 0 else np.nan

        # ATR
        prev_close = close_series.iloc[-2] if n >= 2 else None
        tr = self._update_tr(high=float(h), low=float(l), prev_close=prev_close)
        atr = np.mean(list(self.state.tr_list)[-self.atr_window:]) if len(self.state.tr_list) else np.nan

        # Z-score
        if n >= 1:
            window = self.z_window
            series20 = close_series.iloc[-window:] if n >= window else close_series
            mean20 = series20.mean()
            std20 = series20.std()
            zscore = (close_series.iloc[-1] - mean20) / (std20 if std20 != 0 else np.nan)
        else:
            zscore = np.nan

        # Volume spike
        vol_hist = list(self.state.vols)
        if len(vol_hist):
            vol_avg = np.mean(vol_hist[-self.vol_window:]) if len(vol_hist) else np.nan
            volume_spike = float(v) / (vol_avg if vol_avg != 0 else np.nan)
        else:
            volume_spike = np.nan

        # Price range ratio
        price_range_ratio = (float(h) - float(l)) / (float(o) if float(o) != 0 else np.nan)

        # Volatility regime
        atr_list = list(self.state.tr_list)
        if len(atr_list) >= 1:
            atr_med = np.median(atr_list[-50:]) if len(atr_list) else np.nan
            vol_regime = 1 if atr > (1.0 * atr_med if not np.isnan(atr_med) else np.inf) else 0
        else:
            vol_regime = np.nan

        return {
            "ret_1m": ret_1m,
            "ret_5m": ret_5m,
            "ret_15m": ret_15m,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_trend": ema_trend,
            "ema_trend_norm": ema_trend_norm,
            "rolling_std_5": rolling_std_5,
            "atr": atr,
            "zscore": zscore,
            "volume_spike": volume_spike,
            "price_range_ratio": price_range_ratio,
            "vol_regime": vol_regime,
        }
