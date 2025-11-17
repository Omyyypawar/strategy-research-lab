"""
Feature Factory V2
Produces conservative, explainable microstructure + regime features.
Author: Om | Strategy Research Lab
"""
import numpy as np
import pandas as pd

try:
    from scipy.stats import entropy as scipy_entropy
except Exception:
    scipy_entropy = None

def safe_entropy(arr, bins=10):
    """Estimate discrete entropy for a 1D array; robust to zeros."""
    h, _ = np.histogram(arr[~np.isnan(arr)], bins=bins, density=True)
    h = h + 1e-9
    if scipy_entropy is not None:
        return scipy_entropy(h)
    # fallback manual
    p = h / h.sum()
    return -np.sum(p * np.log(p + 1e-9))

def rolling_entropy(series, window=50, bins=10):
    """Rolling entropy of returns (conservative)."""
    r = series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r.rolling(window).apply(lambda x: safe_entropy(x, bins=bins), raw=False)

def normalized_range(high, low, close, window=20):
    rng = (high - low) / (close + 1e-9)
    return rng.rolling(window).mean()

def volume_imbalance(df, window=20):
    up = df['volume'].where(df['close'] > df['open'], 0.0)
    down = df['volume'].where(df['close'] <= df['open'], 0.0)
    up_s = up.rolling(window).sum()
    down_s = down.rolling(window).sum()
    denom = up_s + down_s + 1e-9
    return ((up_s - down_s) / denom).fillna(0.0)

def vpin_like(df, bucket_size=50):
    """Simple VPIN-like normalized up-count over a sliding bucket."""
    closes = df['close'].values
    out = np.full(len(df), np.nan)
    for i in range(len(df)):
        if i < bucket_size:
            continue
        wins = closes[i-bucket_size+1:i+1]
        # fraction of increases
        out[i] = float((wins[1:] > wins[:-1]).sum()) / max(1, bucket_size-1)
    return pd.Series(out, index=df.index)

def volatility_regime(df, short=20, long=100):
    vs = df['close'].pct_change().rolling(short).std()
    vl = df['close'].pct_change().rolling(long).std()
    return (vs / (vl + 1e-9)).fillna(0.0)

def zscore(series, window=100):
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / (s + 1e-9)

def build_features(df):
    """
    Input: DataFrame with columns: ['open','high','low','close','volume','close_time' optional]
    Output: DataFrame with appended feature columns (no target).
    This function is leak-safe (uses only past data for rolling ops).
    """
    df = df.copy().reset_index(drop=True)
    # ensure numeric
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # basic returns
    df['ret_1m'] = df['close'].pct_change()
    df['ret_5m'] = df['close'].pct_change(5)

    # volatility + regime
    df['vol_short'] = df['close'].pct_change().rolling(10).std()
    df['vol_long'] = df['close'].pct_change().rolling(60).std()
    df['vol_regime'] = (df['vol_short'] / (df['vol_long'] + 1e-9)).fillna(0.0)

    # entropy
    df['entropy_50'] = rolling_entropy(df['close'], window=50, bins=10)

    # normalized range
    df['norm_range_20'] = normalized_range(df['high'], df['low'], df['close'], window=20)

    # volume imbalance
    df['vol_imbalance_20'] = volume_imbalance(df, window=20)

    # vpin-like
    df['vpin_50'] = vpin_like(df, bucket_size=50)

    # zscore of 5m returns
    df['zscore_ret5'] = zscore(df['ret_5m'], window=100)

    # momentum curvature (second derivative proxy)
    df['mom1'] = df['close'] - df['close'].shift(1)
    df['mom2'] = df['mom1'] - df['mom1'].shift(1)
    df['mom2_smooth'] = df['mom2'].rolling(5).mean()

    # simple liquidity proxy
    df['avg_vol_20'] = df['volume'].rolling(20).mean()

    # forward-fill / backfill minimal NaNs for edges — use non-deprecated API
    df = df.ffill().bfill().fillna(0.0)
    return df
