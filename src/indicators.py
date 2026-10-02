"""
Phase 2 Indicators Module  |  Crypto + Equity Compatible
Includes SMA, EMA, RSI, MACD, SmartMomentum, Bollinger Bands
"""

import pandas as pd
import numpy as np

# --------------------------------------------------------------
def SMA(series: pd.Series, period: int = 14) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()

def EMA(series: pd.Series, period: int = 14) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan) + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def MACD(series: pd.Series, short: int = 12, long: int = 26, signal: int = 9):
    short_ema = EMA(series, short)
    long_ema = EMA(series, long)
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def SmartMomentum(series: pd.Series, rsi_period: int = 14, ema_period: int = 9) -> pd.Series:
    series = pd.Series(series).copy()
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    volatility = series.pct_change().rolling(window=20, min_periods=1).std().fillna(0)
    adaptive = np.clip(1 - (volatility / (volatility.max() + 1e-9)), 0.3, 1.0)
    smooth, prev = [], rsi.iloc[0]
    for i in range(len(rsi)):
        alpha = 2.0 / ((ema_period * adaptive.iloc[i]) + 1.0)
        prev = alpha * rsi.iloc[i] + (1 - alpha) * prev
        smooth.append(prev)
    s = pd.Series(smooth, index=rsi.index)
    return ((s - s.min()) / (s.max() - s.min() + 1e-9)).clip(0, 1)

def Bollinger_Bands(series, period=20, std_dev=2):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    break_up = series > upper
    break_down = series < lower
    signal = break_up.astype(int) - break_down.astype(int)
    return upper, mid, lower, signal

def compute_indicators(df: pd.DataFrame):
    close = df['Close']
    df['SMA_20'] = SMA(close, 20)
    df['EMA_20'] = EMA(close, 20)
    df['RSI'] = RSI(close, 14)
    m_line, m_sig, m_hist = MACD(close)
    df['MACD_Line'], df['MACD_Signal'], df['MACD_Hist'] = m_line, m_sig, m_hist
    df['SmartMom'] = SmartMomentum(close)
    df['BB_Upper'], df['BB_Mid'], df['BB_Lower'], df['BB_Signal'] = Bollinger_Bands(close)
    return df



