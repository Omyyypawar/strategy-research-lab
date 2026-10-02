"""
Phase 2 | Step 1 — Feature Factory (Legacy / Batch / Backtest)
Target horizon: 5 minutes (10 samples @ 30s)
Author: Om | Strategy Research Lab

DEPRECATED: This module is superseded by feature_factory_v2.py which provides
19 microstructure + regime features used in production ML pipeline.
Kept for backward compatibility with Phase 2 hybrid strategy only.
Use src.feature_factory_v2.build_features() for all new work.
"""

import numpy as np
import pandas as pd


# -------------------------
# Utility helpers
# -------------------------
def _check_required_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required cols: {missing}")


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


# -------------------------
# EMA helper (pandas ewm)
# -------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# -------------------------
# Primary batch feature builder
# -------------------------
def build_feature_matrix(
    df: pd.DataFrame,
    sampling_seconds: int = 30,
    horizon_minutes: int = 5,
    atr_window: int = 14,
    z_window: int = 20,
    vol_window: int = 20,
    debug: bool = False,
):
    """
    Build quant-grade features for a 5-minute prediction horizon.
    - df must contain columns: ['open','high','low','close','volume'] (lowercase).
    - sampling_seconds default 30s.
    - horizon_minutes default 5 -> prediction horizon_steps = horizon_minutes * 60 / sampling_seconds
    """

    required = ["open", "high", "low", "close", "volume"]
    _check_required_columns(df, required)
    df = df.copy()

    # ensure numeric
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    steps = int(horizon_minutes * 60 / sampling_seconds)  # for 5min @30s => 10
    if steps < 1:
        raise ValueError("Horizon too small relative to sampling_seconds")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # -------------------------
    # 1) Returns at multiple horizons (no future leakage)
    # -------------------------
    # Note: These are backward-looking returns (features). Label uses forward shift.
    df["ret_1m"] = close.pct_change(periods=int(60 / sampling_seconds))   # 1 minute
    df["ret_5m"] = close.pct_change(periods=steps)                       # 5 minutes
    df["ret_15m"] = close.pct_change(periods=int(15 * 60 / sampling_seconds))

    # -------------------------
    # 2) EMA trend (fast - slow)
    # -------------------------
    df["ema_fast"] = ema(close, span=8)
    df["ema_slow"] = ema(close, span=21)
    df["ema_trend"] = df["ema_fast"] - df["ema_slow"]
    df["ema_trend_norm"] = _safe_div(df["ema_trend"], close)

    # -------------------------
    # 3) Rolling volatility (short-window noise estimator)
    # -------------------------
    df["rolling_std_5"] = close.rolling(window=5, min_periods=1).std()

    # -------------------------
    # 4) ATR (True Range) & rolling ATR
    # -------------------------
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["tr"] = tr
    df["atr"] = tr.rolling(window=atr_window, min_periods=1).mean()

    # -------------------------
    # 5) Z-score (mean reversion)
    # -------------------------
    mean_z = close.rolling(window=z_window, min_periods=1).mean()
    std_z = close.rolling(window=z_window, min_periods=1).std().replace(0, np.nan)
    df["zscore"] = (close - mean_z) / std_z

    # -------------------------
    # 6) Volume spike
    # -------------------------
    df["vol_avg"] = vol.rolling(window=vol_window, min_periods=1).mean()
    df["volume_spike"] = _safe_div(vol, df["vol_avg"])

    # -------------------------
    # 7) Price range ratio (intracandle)
    # -------------------------
    df["price_range_ratio"] = _safe_div(high - low, df["open"])

    # -------------------------
    # 8) Volatility regime (binary)
    #    Compare current ATR to rolling median ATR (robust)
    # -------------------------
    df["atr_median_50"] = df["atr"].rolling(window=50, min_periods=1).median()
    df["vol_regime"] = np.where(df["atr"] > df["atr_median_50"], 1, 0)

    # -------------------------
    # 9) Derived combo features (interaction terms)
    #    Keep simple low-dim interactions only, avoids explosion
    # -------------------------
    df["trend_times_vol"] = df["ema_trend_norm"] * df["rolling_std_5"]
    df["zscore_times_volspike"] = df["zscore"] * df["volume_spike"]

    # -------------------------
    # 10) Target: 5-minute forward return & binary target
    #    IMPORTANT: this creates labels only, features above are strictly lagged or contemporaneous
    # -------------------------
    df["future_ret_5m"] = close.shift(-steps) / close - 1.0
    df["target_bin"] = (df["future_ret_5m"] > 0).astype(int)

    # -------------------------
    # 11) Clean & safe fill (no forward-fill that introduces leakage into features)
    #    We will:
    #      - forward/backfill only purely technical indicator NaNs that arise from ewm/rolling
    #      - keep initial NaNs (for training we will drop rows where target or key features missing)
    # -------------------------
    # indicators safe fill: ewm/rolling created NaNs only at the top; fill those with bfill for stability
    indicator_cols = [
        "ema_fast", "ema_slow", "ema_trend", "ema_trend_norm",
        "rolling_std_5", "atr", "zscore", "vol_avg", "volume_spike",
        "price_range_ratio", "atr_median_50", "vol_regime",
        "trend_times_vol", "zscore_times_volspike"
    ]
    df[indicator_cols] = df[indicator_cols].fillna(method="bfill").fillna(method="ffill")

    # -------------------------
    # 12) Final column list (explicit)
    # -------------------------
    feature_cols = [
        "ret_1m", "ret_5m", "ret_15m",
        "ema_fast", "ema_slow", "ema_trend", "ema_trend_norm",
        "rolling_std_5", "atr",
        "zscore", "volume_spike", "price_range_ratio",
        "vol_regime",
        "trend_times_vol", "zscore_times_volspike"
    ]

    # Keep original OHLCV for diagnostics
    output_cols = ["open", "high", "low", "close", "volume"] + feature_cols + ["future_ret_5m", "target_bin"]

    # Reindex to ensure deterministic column order and return a copy
    out = df.reindex(columns=[c for c in output_cols if c in df.columns]).copy()

    if debug:
        print("Feature matrix debug:")
        print(out[feature_cols].describe().T)

    return out

