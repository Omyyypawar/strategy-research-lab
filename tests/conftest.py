"""Shared test fixtures for Strategy Research Lab."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv():
    """Generate 200 rows of realistic OHLCV data for testing."""
    np.random.seed(42)
    n = 200
    base_price = 90000.0
    returns = np.random.normal(0, 0.002, n)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.001, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.001, n)))
    open_ = close * (1 + np.random.normal(0, 0.0005, n))
    volume = np.random.uniform(100, 5000, n)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def sample_ohlcv_small():
    """Minimal OHLCV (10 rows) for edge-case tests."""
    np.random.seed(99)
    n = 10
    close = np.linspace(100, 110, n)
    return pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.random.uniform(50, 200, n),
    })
