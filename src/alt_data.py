"""
Alternative Data Fetcher — Strategy Research Lab
=================================================
Fetches on-chain and derivatives data that most retail bots ignore:
- Funding rates (Binance Futures)
- Open interest (Binance Futures)
- Liquidation events (Binance WebSocket)
- Long/short ratio

These features have proven alpha (Aloosh & Bekaert 2022) and are FREE
from the same Binance API the project already uses.

Author: Om | Strategy Research Lab
"""

import os
import asyncio
import logging
import time
import numpy as np
import pandas as pd
from collections import deque
from typing import Optional

logger = logging.getLogger("alt_data")


# ------------------------------------------------------------------
# Sync fetchers (for batch / dataset building)
# ------------------------------------------------------------------
def fetch_funding_rates(client, symbol: str = "BTCUSDT", limit: int = 500) -> pd.DataFrame:
    """
    Fetch historical funding rates from Binance Futures.
    Uses: GET /fapi/v1/fundingRate
    Returns DataFrame with columns: [funding_time, funding_rate]
    """
    try:
        rates = client.futures_funding_rate(symbol=symbol, limit=limit)
        if not rates:
            return pd.DataFrame(columns=["funding_time", "funding_rate"])
        df = pd.DataFrame(rates)
        df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["funding_rate"] = df["fundingRate"].astype(float)
        return df[["funding_time", "funding_rate"]]
    except Exception as e:
        logger.warning("Failed to fetch funding rates: %s", e)
        return pd.DataFrame(columns=["funding_time", "funding_rate"])


def fetch_open_interest_hist(client, symbol: str = "BTCUSDT", period: str = "5m", limit: int = 500) -> pd.DataFrame:
    """
    Fetch historical open interest from Binance Futures.
    Uses: GET /futures/data/openInterestHist
    Returns DataFrame with columns: [oi_time, open_interest, open_interest_value]
    """
    try:
        data = client.futures_open_interest_hist(symbol=symbol, period=period, limit=limit)
        if not data:
            return pd.DataFrame(columns=["oi_time", "open_interest", "open_interest_value"])
        df = pd.DataFrame(data)
        df["oi_time"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open_interest"] = df["sumOpenInterest"].astype(float)
        df["open_interest_value"] = df["sumOpenInterestValue"].astype(float)
        return df[["oi_time", "open_interest", "open_interest_value"]]
    except Exception as e:
        logger.warning("Failed to fetch OI history: %s", e)
        return pd.DataFrame(columns=["oi_time", "open_interest", "open_interest_value"])


def fetch_long_short_ratio(client, symbol: str = "BTCUSDT", period: str = "5m", limit: int = 500) -> pd.DataFrame:
    """
    Fetch top trader long/short ratio from Binance Futures.
    Uses: GET /futures/data/topLongShortAccountRatio
    """
    try:
        data = client.futures_top_longshort_account_ratio(symbol=symbol, period=period, limit=limit)
        if not data:
            return pd.DataFrame(columns=["ls_time", "long_short_ratio", "long_account", "short_account"])
        df = pd.DataFrame(data)
        df["ls_time"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["long_short_ratio"] = df["longShortRatio"].astype(float)
        df["long_account"] = df["longAccount"].astype(float)
        df["short_account"] = df["shortAccount"].astype(float)
        return df[["ls_time", "long_short_ratio", "long_account", "short_account"]]
    except Exception as e:
        logger.warning("Failed to fetch L/S ratio: %s", e)
        return pd.DataFrame(columns=["ls_time", "long_short_ratio", "long_account", "short_account"])


# ------------------------------------------------------------------
# Async fetchers (for live executors)
# ------------------------------------------------------------------
async def async_fetch_funding_rate(client, symbol: str = "BTCUSDT") -> Optional[float]:
    """Fetch latest funding rate asynchronously."""
    try:
        rates = await client.futures_funding_rate(symbol=symbol, limit=1)
        if rates:
            return float(rates[-1]["fundingRate"])
    except Exception as e:
        logger.debug("Async funding rate fetch failed: %s", e)
    return None


async def async_fetch_open_interest(client, symbol: str = "BTCUSDT") -> Optional[float]:
    """Fetch current open interest asynchronously."""
    try:
        oi = await client.futures_open_interest(symbol=symbol)
        if oi:
            return float(oi["openInterest"])
    except Exception as e:
        logger.debug("Async OI fetch failed: %s", e)
    return None


# ------------------------------------------------------------------
# Rolling alternative data state (for live streaming)
# ------------------------------------------------------------------
class AltDataState:
    """
    Maintains rolling buffers for alternative data features.
    Called by live executors alongside candle processing.
    """

    def __init__(self, maxlen: int = 500):
        self.funding_rates = deque(maxlen=maxlen)
        self.open_interest = deque(maxlen=maxlen)
        self.liquidation_counts = deque(maxlen=maxlen)
        self.long_short_ratios = deque(maxlen=maxlen)

        # liquidation tracking (count per interval)
        self._liq_count_current = 0
        self._liq_volume_current = 0.0

    def append_funding_rate(self, rate: float):
        self.funding_rates.append(rate)

    def append_open_interest(self, oi: float):
        self.open_interest.append(oi)

    def append_long_short_ratio(self, ratio: float):
        self.long_short_ratios.append(ratio)

    def record_liquidation(self, qty: float = 1.0):
        """Call each time a liquidation event is received from WebSocket."""
        self._liq_count_current += 1
        self._liq_volume_current += qty

    def flush_liquidations(self):
        """Call at the end of each candle interval to snapshot liquidation count."""
        self.liquidation_counts.append(self._liq_count_current)
        self._liq_count_current = 0
        self._liq_volume_current = 0.0

    def compute_features(self) -> dict:
        """
        Compute alternative data features from rolling buffers.
        Returns dict of features to merge into the main feature row.
        """
        feats = {}

        # Funding rate features
        if len(self.funding_rates) > 0:
            fr_arr = np.array(self.funding_rates)
            feats["funding_rate"] = fr_arr[-1]
            feats["funding_rate_ma5"] = np.mean(fr_arr[-5:]) if len(fr_arr) >= 5 else fr_arr.mean()
            # extreme funding = contrarian signal
            feats["funding_rate_zscore"] = (
                (fr_arr[-1] - fr_arr.mean()) / (fr_arr.std() + 1e-9)
            ) if len(fr_arr) >= 10 else 0.0
        else:
            feats["funding_rate"] = 0.0
            feats["funding_rate_ma5"] = 0.0
            feats["funding_rate_zscore"] = 0.0

        # Open interest features
        if len(self.open_interest) > 0:
            oi_arr = np.array(self.open_interest)
            feats["open_interest"] = oi_arr[-1]
            # OI change rate (pct change from 5 periods ago)
            if len(oi_arr) >= 6:
                feats["oi_change_5"] = (oi_arr[-1] / (oi_arr[-6] + 1e-9)) - 1.0
            else:
                feats["oi_change_5"] = 0.0
            # OI divergence from rolling mean (signals crowding)
            feats["oi_zscore"] = (
                (oi_arr[-1] - oi_arr.mean()) / (oi_arr.std() + 1e-9)
            ) if len(oi_arr) >= 10 else 0.0
        else:
            feats["open_interest"] = 0.0
            feats["oi_change_5"] = 0.0
            feats["oi_zscore"] = 0.0

        # Liquidation features
        if len(self.liquidation_counts) > 0:
            liq_arr = np.array(self.liquidation_counts)
            feats["liq_count"] = liq_arr[-1]
            feats["liq_count_ma10"] = np.mean(liq_arr[-10:]) if len(liq_arr) >= 10 else liq_arr.mean()
            # liquidation spike = high short-term volatility signal
            feats["liq_spike"] = (
                liq_arr[-1] / (feats["liq_count_ma10"] + 1e-9)
            )
        else:
            feats["liq_count"] = 0
            feats["liq_count_ma10"] = 0.0
            feats["liq_spike"] = 0.0

        # Long/short ratio features
        if len(self.long_short_ratios) > 0:
            ls_arr = np.array(self.long_short_ratios)
            feats["long_short_ratio"] = ls_arr[-1]
            # extreme ratio = contrarian signal
            feats["ls_ratio_zscore"] = (
                (ls_arr[-1] - ls_arr.mean()) / (ls_arr.std() + 1e-9)
            ) if len(ls_arr) >= 10 else 0.0
        else:
            feats["long_short_ratio"] = 1.0
            feats["ls_ratio_zscore"] = 0.0

        return feats


# ------------------------------------------------------------------
# Feature names for manifest
# ------------------------------------------------------------------
ALT_DATA_FEATURE_NAMES = [
    "funding_rate", "funding_rate_ma5", "funding_rate_zscore",
    "open_interest", "oi_change_5", "oi_zscore",
    "liq_count", "liq_count_ma10", "liq_spike",
    "long_short_ratio", "ls_ratio_zscore",
]
