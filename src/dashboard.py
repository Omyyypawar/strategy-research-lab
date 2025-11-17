"""
Streamlit dashboard for Strategy Research Lab
Phase 5 — Live Control & Monitoring UI for Hybrid + ML models

Save to: src/dashboard.py
Run:  streamlit run src/dashboard.py   (from project root)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import subprocess
import os
from datetime import datetime

st.set_page_config(layout="wide", page_title="Strategy Research Lab Dashboard")

# ---------------------------
# Utility functions
# ---------------------------
def calculate_metrics(df: pd.DataFrame) -> dict:
    """Compute CAGR, Sharpe, Sortino, Max Drawdown, Final Equity."""
    metrics = {"CAGR (%)": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "Max Drawdown (%)": np.nan, "Final Equity": np.nan}
    if df is None or df.empty:
        return metrics

    if "Return" not in df.columns:
        df["Return"] = df["Close"].pct_change().fillna(0)

    if "Equity_Curve" not in df.columns:
        if "Strategy_Return" in df.columns:
            df["Equity_Curve"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()
        else:
            df["Equity_Curve"] = (1 + df["Return"].fillna(0)).cumprod()

    total_days = len(df)
    total_years = total_days / 252 if total_days > 0 else np.nan
    ending = df["Equity_Curve"].iloc[-1] if len(df) else np.nan
    if pd.notna(ending) and total_years and total_years > 0:
        cagr = ending ** (1 / total_years) - 1
    else:
        cagr = np.nan

    daily_mean = df["Return"].mean()
    daily_std = df["Return"].std() + 1e-12
    sharpe = np.sqrt(252) * daily_mean / daily_std if daily_std > 0 else np.nan
    neg_std = df.loc[df["Return"] < 0, "Return"].std() + 1e-12
    sortino = np.sqrt(252) * daily_mean / neg_std if neg_std > 0 else np.nan
    cum = df["Equity_Curve"].cummax()
    drawdown = (cum - df["Equity_Curve"]) / cum
    max_dd = drawdown.max()

    metrics["CAGR (%)"] = round(cagr * 100, 2) if pd.notna(cagr) else np.nan
    metrics["Sharpe"] = round(sharpe, 2) if pd.notna(sharpe) else np.nan
    metrics["Sortino"] = round(sortino, 2) if pd.notna(sortino) else np.nan
    metrics["Max Drawdown (%)"] = round(max_dd * 100, 2) if pd.notna(max_dd) else np.nan
    metrics["Final Equity"] = round(ending, 3) if pd.notna(ending) else np.nan

    return metrics


def load_csv_safe(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    except Exception:
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
    return df


def draw_equity_plot(hybrid_df, ml_df, ticker="BTC-USD"):
    fig = go.Figure()
    if hybrid_df is not None and "Equity_Curve" in hybrid_df.columns:
        fig.add_trace(go.Scatter(x=hybrid_df.index, y=hybrid_df["Equity_Curve"], name="Hybrid Equity", line=dict(color="cyan")))
    if ml_df is not None and "Equity_Curve" in ml_df.columns:
        fig.add_trace(go.Scatter(x=ml_df.index, y=ml_df["Equity_Curve"], name="ML Equity", line=dict(color="orange")))
    fig.update_layout(template="plotly_dark", title=f"{ticker} Equity Curve Comparison", height=420)
    return fig


def draw_rolling_sharpe(hybrid_df, ml_df, window=126):
    fig = go.Figure()
    if hybrid_df is not None and "Return" in hybrid_df.columns:
        r = hybrid_df["Return"].fillna(0)
        rm = r.rolling(window).mean()
        rs = r.rolling(window).std().replace(0, np.nan)
        rolling_sharpe = np.sqrt(252) * rm / rs
        fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name="Hybrid Rolling Sharpe", line=dict(color="lightcyan")))
    if ml_df is not None and "Return" in ml_df.columns:
        r2 = ml_df["Return"].fillna(0)
        rm2 = r2.rolling(window).mean()
        rs2 = r2.rolling(window).std().replace(0, np.nan)
        rolling_sharpe2 = np.sqrt(252) * rm2 / rs2
        fig.add_trace(go.Scatter(x=rolling_sharpe2.index, y=rolling_sharpe2, name="ML Rolling Sharpe", line=dict(color="orange")))
    fig.update_layout(template="plotly_dark", title=f"Rolling Sharpe (window={window})", height=300)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig


def show_feature_importance(model_path, phase2_path, top_n=15):
    if not os.path.exists(model_path):
        st.warning("Model file not found for feature importances.")
        return
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return

    df_phase2 = load_csv_safe(phase2_path)
    if df_phase2 is None:
        st.warning("phase2 features file not found; cannot show feature names reliably.")
        return

    numeric_cols = df_phase2.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["Target", "Return", "Strategy_Return", "Equity_Curve"]
    features = [c for c in numeric_cols if c not in exclude]

    importances = None
    if hasattr(model, "feature_importances_"):
        try:
            fi = model.feature_importances_
            if len(fi) == len(features):
                importances = pd.Series(fi, index=features).sort_values(ascending=False)
            else:
                importances = pd.Series(fi[:len(features)], index=features[:len(fi)]).sort_values(ascending=False)
        except Exception:
            importances = None

    if importances is None:
        st.warning("Model feature importances unavailable; showing correlation fallback.")
        corr = df_phase2[features + ["Target"]].corr()["Target"].abs().sort_values(ascending=False)
        importances = corr.drop("Target", errors="ignore")

    st.bar_chart(importances.head(top_n))


# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("🧠 Strategy Control Panel")

if st.sidebar.button("🏁 Run Full Pipeline"):
    st.sidebar.write("⚙️ Running full pipeline (main.py)... please wait...")
    os.system("python src/main.py")
    st.sidebar.success("✅ Pipeline completed successfully!")

if st.sidebar.button("🤖 Retrain ML Model"):
    st.sidebar.write("🧩 Training ML model...")
    os.system("python src/ml_signal_engine.py")
    st.sidebar.success("✅ Model retraining complete!")

if st.sidebar.button("📊 Compare ML vs Hybrid"):
    st.sidebar.write("📈 Launching comparison dashboard...")
    os.system("python src/comparison_dashboard.py")

if st.sidebar.button("💹 Launch Live Paper Trader"):
    st.sidebar.write("🚀 Starting live trading simulation...")
    os.system("python src/live_trading.py")


# ---------------------------
# Main Dashboard Layout
# ---------------------------
st.title("📊 Strategy Research Lab — Live Dashboard")
st.markdown("Hybrid (Momentum + Mean Reversion) + ML comparison. Control panel & quick actions for retraining and running pipeline.")

ticker = "BTC-USD"
st.info(f"Showing metrics for: **{ticker}**")

hybrid_file = "results_BTC-USD_phase2_step1.csv"
ml_file = "phase3_ML_signals_BTC.csv"
phase2_file = "phase2_features_BTC.csv"
model_file = "phase3_rf_model.pkl"

hybrid_df = load_csv_safe(hybrid_file)
ml_df = load_csv_safe(ml_file)

# Top metrics
st.subheader("Performance summary (Hybrid vs ML)")
hybrid_metrics = calculate_metrics(hybrid_df)
ml_metrics = calculate_metrics(ml_df)
metrics_table = pd.DataFrame([hybrid_metrics, ml_metrics], index=["Hybrid", "ML"])
st.dataframe(metrics_table.T)

# Equity plot
st.subheader("Equity Curves")
st.plotly_chart(draw_equity_plot(hybrid_df, ml_df, ticker=ticker), use_container_width=True)

# Rolling sharpe
st.subheader("Rolling Sharpe")
st.plotly_chart(draw_rolling_sharpe(hybrid_df, ml_df, window=126), use_container_width=True)

# Feature importance
st.subheader("Feature Importances (Model)")
show_feature_importance(model_file, phase2_file, top_n=15)

# Snapshots
st.subheader("Latest Data Snapshots")
col1, col2 = st.columns(2)
with col1:
    st.write("Hybrid last 5 rows")
    st.dataframe(hybrid_df.tail(5) if hybrid_df is not None else pd.DataFrame())

with col2:
    st.write("ML last 5 rows")
    st.dataframe(ml_df.tail(5) if ml_df is not None else pd.DataFrame())

st.markdown("---")
st.caption("Tip: Use the 'Retrain ML' button after running full pipeline if you updated feature generation.")
