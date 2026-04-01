# src/live_dashboard.py
"""
Strategy Research Lab — Live Dashboard (Institutional theme)
Author: Om Pawar
Theme: Clean white / navy (Investor-ready)
Compatible with executor_v4_log.csv (executor_v4 / live_executor)
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import plotly.graph_objects as go
from datetime import datetime

# -------------------------
# Page config + Theme CSS
# -------------------------
st.set_page_config(page_title="Strategy Research Lab — Live Monitor", layout="wide")

# Institutional color palette
NAVY = "#1B2A49"
ACCENT = "#017BFE"
BUY_GREEN = "#0BA360"
SELL_RED = "#D9534F"
BG = "#F7F9FC"
TEXT = "#222222"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#E0E6ED"

st.markdown(
    f"""
    <style>
    /* page background */
    .reportview-container .main{{background-color: {BG};}}
    .css-1l02zno {{background-color: {BG};}}
    /* headings + text */
    .stApp {{ color: {TEXT}; font-family: "Segoe UI", Roboto, Arial, sans-serif; }}
    /* metric tiles */
    .metric-box {{ background: {CARD_BG}; border-radius: 10px; padding: 12px; border: 1px solid {CARD_BORDER}; }}
    .small-muted {{ color: #6b7280; font-size:12px; }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Config
# -------------------------
LOG_FILE = os.getenv("LOG_FILE", "executor_v4_log.csv")
AUTO_REFRESH_DEFAULT =  True
REFRESH_DEFAULT = 10  # seconds

# -------------------------
# Helpers
# -------------------------
def safe_load_csv(path: str) -> pd.DataFrame:
    """Load CSV defensively (handles concurrent writes)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    for _ in range(3):
        try:
            df = pd.read_csv(path)
            if df.empty:
                return pd.DataFrame()
            # Try parse timestamp-like fields
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            # ensure important numeric columns exist
            for col in ["equity", "close", "signal", "action", "signal_prob"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception:
            time.sleep(0.2)
    return pd.DataFrame()

def calc_metrics(df: pd.DataFrame) -> dict:
    """Return dictionary of key performance metrics (robust)."""
    if df.empty or "equity" not in df.columns or len(df) < 2:
        return {"FinalEq": 0.0, "CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0, "Trades": 0}
    eq = df["equity"].fillna(method="ffill").fillna(0.0)
    # returns
    rets = eq.pct_change().fillna(0.0)
    # approximate CAGR (uses rows as periods; assumes 1 trading day per row is NOT true — this is a quick proxy)
    n = max(len(eq), 1)
    try:
        cagr = (eq.iloc[-1] / max(eq.iloc[0], 1e-9)) ** (252.0 / n) - 1.0
    except Exception:
        cagr = 0.0
    sharpe = np.sqrt(252) * rets.mean() / (rets.std() + 1e-9)
    drawdown = 1.0 - eq / eq.cummax()
    maxdd = drawdown.max()
    trades = 0
    if "action" in df.columns:
        trades = int(df["action"].isin(["BUY", "SELL"]).sum())
    return {
        "FinalEq": round(eq.iloc[-1], 4),
        "CAGR": round(cagr * 100, 2),
        "Sharpe": round(sharpe, 3),
        "MaxDD": round(maxdd * 100, 2),
        "Trades": trades
    }

def ai_commentary(df: pd.DataFrame, metrics: dict) -> str:
    """Small deterministic AI-like commentary (no external calls)."""
    if df.empty:
        return "📭 Waiting for live trading data..."
    last_action = df["action"].iloc[-1] if "action" in df.columns else "HOLD"
    comments = []
    if last_action == "BUY":
        comments.append("🟢 Recent action: BUY — model shows bullish short-term signal.")
    elif last_action == "SELL":
        comments.append("🔴 Recent action: SELL — model closed position / realized PnL.")
    else:
        comments.append("⚪ Model holding — no immediate directional action.")

    # Metric-driven notes
    cagr = metrics.get("CAGR", 0.0)
    sharpe = metrics.get("Sharpe", 0.0)
    maxdd = metrics.get("MaxDD", 0.0)
    if cagr > 5:
        comments.append(f"📈 Equity trending up (CAGR ≈ {cagr}%).")
    elif cagr < 0:
        comments.append(f"📉 Small negative drift detected (CAGR {cagr}%).")
    else:
        comments.append("⚖️ Equity stable across the test window.")

    if sharpe > 1.0:
        comments.append("🚀 Sharpe healthy: risk is being rewarded.")
    elif sharpe > 0.3:
        comments.append("🧩 Sharpe moderate—system may benefit from noise-reduction.")
    else:
        comments.append("😴 Sharpe low — prioritize volatility / regime filters next.")

    if maxdd > 10:
        comments.append("⚠️ Drawdown material — implement drawdown stop / dynamic sizing.")
    else:
        comments.append("🛡️ Drawdown within acceptable range for alpha exploration.")

    comments.append(f"📊 Total trades: **{metrics.get('Trades', 0)}**")
    return "\n\n".join(comments)

# -------------------------
# UI: Header / Controls
# -------------------------
st.title("⚡ Strategy Research Lab — Live Monitor (Institutional)")
st.markdown("**Live paper-trading monitor — investor-ready visuals & interpretability.**")

col_left, col_mid, col_right = st.columns([1.2, 1, 0.8])
auto_refresh = col_left.toggle("🔁 Auto refresh", value=AUTO_REFRESH_DEFAULT)
refresh_sec = col_mid.slider("Refresh interval (sec)", min_value=5, max_value=60, value=REFRESH_DEFAULT, step=1)
st.markdown("---")

# placeholders
metrics_ph = st.empty()
equity_ph = st.empty()
trade_ph = st.empty()
table_ph = st.empty()
comment_ph = st.empty()
raw_ph = st.expander("Raw log preview", expanded=False)

# -------------------------
# Main loop (streaming style)
# -------------------------
while True:
    if not auto_refresh:
        time.sleep(0.2)
        st.info("Auto-refresh paused. Toggle to resume.", icon="⏸️")
        break

    df = safe_load_csv(LOG_FILE)

    if df.empty:
        st.warning("Waiting for executor log file to appear or for data to stream into the CSV...")
        time.sleep(refresh_sec)
        continue

    # Ensure required columns exist
    if "equity" not in df.columns:
        # derive equity if possible
        if "balance" in df.columns and "position" in df.columns and "close" in df.columns:
            df["equity"] = pd.to_numeric(df["balance"], errors="coerce").fillna(0.0) + pd.to_numeric(df["position"], errors="coerce").fillna(0.0) * pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
        else:
            df["equity"] = df.index.map(lambda i: np.nan)

    # compute metrics
    metrics = calc_metrics(df)

    # ---------------- Metrics panel ----------------
    with metrics_ph.container():
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.markdown(f"<div class='metric-box'><h3 style='margin:0'>{metrics['FinalEq']}</h3><div class='small-muted'>Final Equity</div></div>", unsafe_allow_html=True)
        r2.markdown(f"<div class='metric-box'><h3 style='margin:0'>{metrics['CAGR']}%</h3><div class='small-muted'>CAGR (proxy)</div></div>", unsafe_allow_html=True)
        r3.markdown(f"<div class='metric-box'><h3 style='margin:0'>{metrics['Sharpe']}</h3><div class='small-muted'>Sharpe</div></div>", unsafe_allow_html=True)
        r4.markdown(f"<div class='metric-box'><h3 style='margin:0'>{metrics['MaxDD']}%</h3><div class='small-muted'>Max Drawdown</div></div>", unsafe_allow_html=True)
        r5.markdown(f"<div class='metric-box'><h3 style='margin:0'>{metrics['Trades']}</h3><div class='small-muted'>Trades</div></div>", unsafe_allow_html=True)

    # ---------------- Equity chart ----------------
    with equity_ph.container():
        fig = go.Figure()
        # equity
        if "timestamp" in df.columns:
            x = df["timestamp"]
        else:
            x = pd.date_range(end=datetime.utcnow(), periods=len(df), freq="S")

        fig.add_trace(go.Scatter(x=x, y=df["equity"], mode="lines+markers", name="Equity", line=dict(color=ACCENT, width=2)))
        fig.update_layout(template="plotly_white", title="📈 Equity Curve", height=360,
                          paper_bgcolor=BG, plot_bgcolor=BG,
                          margin=dict(t=40,l=10,r=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- Trade / Price chart ----------------
    with trade_ph.container():
        fig2 = go.Figure()
        # price series
        price_col = "close" if "close" in df.columns else (df.columns[0] if len(df.columns)>0 else None)
        if price_col:
            fig2.add_trace(go.Scatter(x=x, y=df[price_col], mode="lines", name="Price", line=dict(color=NAVY)))

        # buys / sells
        if "action" in df.columns and price_col:
            buys = df[df["action"] == "BUY"]
            sells = df[df["action"] == "SELL"]
            if not buys.empty:
                fig2.add_trace(go.Scatter(x=buys["timestamp"] if "timestamp" in buys.columns else buys.index,
                                          y=buys[price_col],
                                          mode="markers", name="BUY",
                                          marker=dict(color=BUY_GREEN, size=10, symbol="triangle-up")))
            if not sells.empty:
                fig2.add_trace(go.Scatter(x=sells["timestamp"] if "timestamp" in sells.columns else sells.index,
                                          y=sells[price_col],
                                          mode="markers", name="SELL",
                                          marker=dict(color=SELL_RED, size=10, symbol="triangle-down")))

        fig2.update_layout(template="plotly_white", title="💹 Price & Executions", height=360,
                           paper_bgcolor=BG, plot_bgcolor=BG,
                           margin=dict(t=40,l=10,r=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- Recent trades table ----------------
    with table_ph.container():
        st.subheader("📜 Recent Trades / Signals")
        show_cols = []
        # preferred columns order if present
        for c in ["timestamp", "action", "close", "signal", "signal_prob", "qty", "entry_price", "exit_price", "pnl"]:
            if c in df.columns:
                show_cols.append(c)
        # fallback: show some useful columns
        if not show_cols:
            show_cols = df.columns[:8].tolist()
        recent = df[show_cols].tail(12).iloc[::-1]
        st.dataframe(recent, use_container_width=True, height=260)

    # ---------------- AI commentary ----------------
    with comment_ph.container():
        st.subheader("🧠 System Commentary")
        st.markdown(f"<div style='background:{CARD_BG}; padding:12px; border-radius:10px; border:1px solid {CARD_BORDER};'>{ai_commentary(df, metrics)}</div>", unsafe_allow_html=True)

    # ---------------- Raw preview (expandable) ----------------
    with raw_ph:
        st.write("Last 20 rows of log (for debugging):")
        st.dataframe(df.tail(20).reset_index(drop=True))

    # footer
    st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Log: {LOG_FILE}")

    time.sleep(refresh_sec)

