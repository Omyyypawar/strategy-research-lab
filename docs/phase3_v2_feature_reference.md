# Phase 3 V2 Feature Reference

This file extracts every feature used by the `phase3_rf_v2` training pipeline.

## Source of truth

- Feature manifest: `models/features_columns_phase3_v2.json`
- Feature builder: `src/feature_factory_v2.py`
- Dataset builder: `src/dataset_builder_v2.py`
- Trainer: `src/experiment_runner_v2.py`

## Exact feature list used by the model

```python
FEATURE_COLUMNS_PHASE3_V2 = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ret_1m",
    "ret_5m",
    "vol_short",
    "vol_long",
    "vol_regime",
    "entropy_50",
    "norm_range_20",
    "vol_imbalance_20",
    "vpin_50",
    "zscore_ret5",
    "mom1",
    "mom2",
    "mom2_smooth",
    "avg_vol_20",
]
```

## Feature code extraction

### Raw market columns

```python
open
high
low
close
volume
```

These come directly from Binance klines in `src/dataset_builder_v2.py`.

### 1) `ret_1m`

```python
df["ret_1m"] = df["close"].pct_change()
```

Meaning: 1-candle return.

### 2) `ret_5m`

```python
df["ret_5m"] = df["close"].pct_change(5)
```

Meaning: 5-candle return.

### 3) `vol_short`

```python
df["vol_short"] = df["close"].pct_change().rolling(10).std()
```

Meaning: short-term rolling volatility over 10 candles.

### 4) `vol_long`

```python
df["vol_long"] = df["close"].pct_change().rolling(60).std()
```

Meaning: long-term rolling volatility over 60 candles.

### 5) `vol_regime`

```python
df["vol_regime"] = (df["vol_short"] / (df["vol_long"] + 1e-9)).fillna(0.0)
```

Meaning: short-vs-long volatility regime ratio.

### 6) `entropy_50`

Helper code:

```python
def safe_entropy(arr, bins=10):
    h, _ = np.histogram(arr[~np.isnan(arr)], bins=bins, density=True)
    h = h + 1e-9
    if scipy_entropy is not None:
        return scipy_entropy(h)
    p = h / h.sum()
    return -np.sum(p * np.log(p + 1e-9))

def rolling_entropy(series, window=50, bins=10):
    r = series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r.rolling(window).apply(lambda x: safe_entropy(x, bins=bins), raw=False)
```

Feature:

```python
df["entropy_50"] = rolling_entropy(df["close"], window=50, bins=10)
```

Meaning: rolling entropy of returns, used as a market randomness/choppiness proxy.

### 7) `norm_range_20`

Helper code:

```python
def normalized_range(high, low, close, window=20):
    rng = (high - low) / (close + 1e-9)
    return rng.rolling(window).mean()
```

Feature:

```python
df["norm_range_20"] = normalized_range(df["high"], df["low"], df["close"], window=20)
```

Meaning: average normalized candle range over 20 candles.

### 8) `vol_imbalance_20`

Helper code:

```python
def volume_imbalance(df, window=20):
    up = df["volume"].where(df["close"] > df["open"], 0.0)
    down = df["volume"].where(df["close"] <= df["open"], 0.0)
    up_s = up.rolling(window).sum()
    down_s = down.rolling(window).sum()
    denom = up_s + down_s + 1e-9
    return ((up_s - down_s) / denom).fillna(0.0)
```

Feature:

```python
df["vol_imbalance_20"] = volume_imbalance(df, window=20)
```

Meaning: net bullish vs bearish volume balance over 20 candles.

### 9) `vpin_50`

Helper code:

```python
def vpin_like(df, bucket_size=50):
    closes = df["close"].values
    out = np.full(len(df), np.nan)
    for i in range(len(df)):
        if i < bucket_size:
            continue
        wins = closes[i-bucket_size+1:i+1]
        out[i] = float((wins[1:] > wins[:-1]).sum()) / max(1, bucket_size-1)
    return pd.Series(out, index=df.index)
```

Feature:

```python
df["vpin_50"] = vpin_like(df, bucket_size=50)
```

Meaning: simple VPIN-like directional imbalance proxy using up-close frequency.

### 10) `zscore_ret5`

Helper code:

```python
def zscore(series, window=100):
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / (s + 1e-9)
```

Feature:

```python
df["zscore_ret5"] = zscore(df["ret_5m"], window=100)
```

Meaning: normalized 5-candle return against its rolling 100-candle history.

### 11) `mom1`

```python
df["mom1"] = df["close"] - df["close"].shift(1)
```

Meaning: first-order momentum.

### 12) `mom2`

```python
df["mom2"] = df["mom1"] - df["mom1"].shift(1)
```

Meaning: second-order momentum / acceleration proxy.

### 13) `mom2_smooth`

```python
df["mom2_smooth"] = df["mom2"].rolling(5).mean()
```

Meaning: smoothed momentum curvature.

### 14) `avg_vol_20`

```python
df["avg_vol_20"] = df["volume"].rolling(20).mean()
```

Meaning: rolling 20-candle average volume.

## Full extracted feature builder code

```python
def build_features(df):
    df = df.copy().reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ret_1m"] = df["close"].pct_change()
    df["ret_5m"] = df["close"].pct_change(5)

    df["vol_short"] = df["close"].pct_change().rolling(10).std()
    df["vol_long"] = df["close"].pct_change().rolling(60).std()
    df["vol_regime"] = (df["vol_short"] / (df["vol_long"] + 1e-9)).fillna(0.0)

    df["entropy_50"] = rolling_entropy(df["close"], window=50, bins=10)
    df["norm_range_20"] = normalized_range(df["high"], df["low"], df["close"], window=20)
    df["vol_imbalance_20"] = volume_imbalance(df, window=20)
    df["vpin_50"] = vpin_like(df, bucket_size=50)
    df["zscore_ret5"] = zscore(df["ret_5m"], window=100)

    df["mom1"] = df["close"] - df["close"].shift(1)
    df["mom2"] = df["mom1"] - df["mom1"].shift(1)
    df["mom2_smooth"] = df["mom2"].rolling(5).mean()

    df["avg_vol_20"] = df["volume"].rolling(20).mean()

    df = df.ffill().bfill().fillna(0.0)
    return df
```

## Training target used in hackathon pipeline

These are not input features, but they were used to create the label:

```python
full["future_close_5m"] = full["close"].shift(-5)
full["future_return_5m"] = (full["future_close_5m"] / full["close"]) - 1
full["target"] = (full["future_return_5m"] > 0).astype(int)
```

Meaning: predict whether price will be up 5 candles ahead.

## Important note for your live-data upgrade

Right now this model uses only Binance candle-derived features.

If you want to include NewsAPI data in the live system, that will require adding new sentiment/news features to the feature builder and retraining the model so training and live inference use the same columns.
