"""
Phase 6.5 — Streamlit Live Dashboard (V2)
Compatible with FeatureFactory V2 + Live Executor V2
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import plotly.graph_objects as go
from datetime import datetime

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
LOG_FILE = "executor_log.csv"   # <--- matches your live_executor.py
REFRESH_SEC = 10

st.set_page_config(page_title="⚡ Strategy Research Lab — Live Dashboard", layout="wide")

# ------------------------------------------------
# SAFE CSV LOADER (handles concurrent writes)
# ------------------------------------------------
def safe_load_csv(path, retries=3):
    if not os.path.exists(path):
        return pd.DataFrame()
    for _ in range(retries):
        try:
            df = pd.read_csv(path)
            if df.empty:
                return pd.DataFrame()
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            return df
        except Exception:
            time.sleep(0.2)
    return pd.DataFrame()

# ------------------------------------------------
# METRIC CALCULATIONS
# ------------------------------------------------
def calc_metrics(df):
    if df.empty or len(df) < 3:
        return {"CAGR": 0, "Sharpe": 0, "MaxDD": 0, "FinalEq": 0, "Trades": 0}

    df = df.copy()
    if "equity" not in df.columns:
        df["equity"] = 0.0
    df["ret"] = df["equity"].pct_change().fillna(0)
    # Use len(df) as proxy for days — approximate but fine for monitoring
    days = max(1, len(df) / (60*24))  # placeholder if extremely high-frequency
    cagr = (df["equity"].iloc[-1] / max(1e-9, df["equity"].iloc[0])) ** (252 / max(1, len(df))) - 1
    sharpe = np.sqrt(252) * df["ret"].mean() / (df["ret"].std() + 1e-9)
    dd = 1 - df["equity"] / df["equity"].cummax()
    maxdd = dd.max()
    trades = int(df["action"].isin(["BUY", "SELL"]).sum())
    return {
        "CAGR": round(cagr * 100, 2),
        "Sharpe": round(sharpe, 2),
        "MaxDD": round(maxdd * 100, 2),
        "FinalEq": round(df["equity"].iloc[-1], 2),
        "Trades": trades
    }

# ------------------------------------------------
# AI Commentary
# ------------------------------------------------
def ai_commentary(df, m):
    if df.empty:
        return "📭 Waiting for live data…"

    comments = []
    last_act = df["action"].iloc[-1] if "action" in df.columns else "HOLD"
    if last_act == "BUY":
        comments.append("🟢 Model executed a **BUY** — possible bullish momentum.")
    elif last_act == "SELL":
        comments.append("🔴 Model executed a **SELL** — potential local weakness.")
    else:
        comments.append("⚪ Model holding — no strong directional signal.")

    if m["CAGR"] > 3:
        comments.append(f"📈 Equity trending up with CAGR **{m['CAGR']}%**.")
    elif m["CAGR"] < 0:
        comments.append(f"📉 Equity slipping (CAGR {m['CAGR']}%).")
    else:
        comments.append(f"⚖️ Stable performance, CAGR {m['CAGR']}%.")

    if m["Sharpe"] > 1:
        comments.append("🚀 Healthy Sharpe — risk is being rewarded.")
    elif m["Sharpe"] > 0.3:
        comments.append("🧩 Moderate Sharpe — range-bound behaviour.")
    else:
        comments.append("😴 Low Sharpe — noise dominating returns.")

    if m["MaxDD"] > 10:
        comments.append("⚠️ High drawdown — model may be overtrading.")
    else:
        comments.append("🛡️ Drawdown controlled and stable.")
    comments.append(f"📊 Total trades so far: **{m['Trades']}**")
    return "\n".join(comments)

# ------------------------------------------------
# UI HEADER
# ------------------------------------------------
st.title("⚡ Strategy Research Lab — Live Trading Dashboard")
col1, col2 = st.columns(2)
auto = col1.toggle("Auto Refresh", value=True)
refresh = col2.slider("Refresh Interval (sec)", 5, 60, REFRESH_SEC)
st.markdown("---")

metric_box = st.empty()
equity_box = st.empty()
trade_box = st.empty()
table_box = st.empty()
comment_box = st.empty()
status_box = st.empty()

# ------------------------------------------------
# MAIN LOOP
# ------------------------------------------------
while True:
    if not auto:
        time.sleep(1)
        continue

    df = safe_load_csv(LOG_FILE)
    if df.empty:
        st.warning("Waiting for live data...")
        time.sleep(refresh)
        continue

    # Ensure columns we rely on exist
    if "close" not in df.columns and "price" in df.columns:
        df = df.rename(columns={"price": "close"})
    if "equity" not in df.columns:
        df["equity"] = df["balance"].fillna(0) + df.get("position", 0).fillna(0) * df.get("close", 0).fillna(0)

    m = calc_metrics(df)

    # METRICS
    with metric_box.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("💰 Final Equity", f"${m['FinalEq']}")
        c2.metric("📈 CAGR", f"{m['CAGR']}%")
        c3.metric("⚡ Sharpe", m["Sharpe"])
        c4.metric("📉 Max DD", f"{m['MaxDD']}%")
        c5.metric("🔁 Trades", m["Trades"])

    # EQUITY CHART
    with equity_box.container():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["equity"], mode="lines+markers", name="Equity", line=dict(color="cyan")))
        fig.update_layout(template="plotly_dark", title="📊 Equity Curve", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # TRADE CHART
    with trade_box.container():
        fig2 = go.Figure()
        if "close" in df.columns:
            fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], mode="lines", name="Price", line=dict(color="white")))
        buys = df[df.get("action") == "BUY"] if "action" in df.columns else pd.DataFrame()
        sells = df[df.get("action") == "SELL"] if "action" in df.columns else pd.DataFrame()
        if not buys.empty:
            fig2.add_trace(go.Scatter(x=buys["timestamp"], y=buys["close"], mode="markers", name="BUY", marker=dict(color="lime", size=10)))
        if not sells.empty:
            fig2.add_trace(go.Scatter(x=sells["timestamp"], y=sells["close"], mode="markers", name="SELL", marker=dict(color="red", size=10)))
        fig2.update_layout(template="plotly_dark", title="💹 Trade Log", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # TABLE
    with table_box.container():
        st.subheader("📜 Recent Trades")
        show_cols = [c for c in ["timestamp", "action", "close", "equity", "signal"] if c in df.columns]
        st.dataframe(df[show_cols].tail(12), use_container_width=True)

    # COMMENTARY
    with comment_box.container():
        st.subheader("🧠 AI Commentary")
        st.markdown(f"<div style='background:#111;padding:12px;border-radius:12px;'>{ai_commentary(df, m)}</div>", unsafe_allow_html=True)

    status_box.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    time.sleep(refresh)

