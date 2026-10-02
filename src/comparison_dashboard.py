"""
Phase 3 · Step 3 — ML vs Hybrid Comparison Dashboard
Loads hybrid & ML outputs, computes metrics, rolling Sharpe, correlation,
and shows an interactive Plotly dashboard.

Save this as src/comparison_dashboard.py
Run standalone: python -m src.comparison_dashboard
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Optional

pio.renderers.default = "browser"


# -------------------------
# Helpers: metrics & rolling sharpe
# -------------------------
def calculate_metrics(df: pd.DataFrame, label: str = "Model") -> dict:
    """Compute CAGR, Sharpe, Sortino, Max Drawdown and Final Equity."""
    df = df.copy()
    # Ensure we have Return and Equity_Curve
    if "Return" not in df.columns:
        df["Return"] = df["Close"].pct_change().fillna(0)
    if "Equity_Curve" not in df.columns:
        # fallback: if Strategy_Return exists
        if "Strategy_Return" in df.columns:
            df["Equity_Curve"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()
        else:
            df["Equity_Curve"] = (1 + df["Return"].fillna(0)).cumprod()

    total_days = len(df)
    total_years = total_days / 252 if total_days > 0 else np.nan
    ending = df["Equity_Curve"].iloc[-1] if len(df) else np.nan
    cagr = (ending ** (1 / total_years) - 1) if (total_years and total_years > 0) else np.nan
    daily_mean = df["Return"].mean()
    daily_std = df["Return"].std() + 1e-12
    sharpe = np.sqrt(252) * daily_mean / daily_std if daily_std > 0 else np.nan
    neg_std = df.loc[df["Return"] < 0, "Return"].std() + 1e-12
    sortino = np.sqrt(252) * daily_mean / neg_std if neg_std > 0 else np.nan

    # max drawdown
    cum = df["Equity_Curve"].cummax()
    drawdown = (cum - df["Equity_Curve"]) / cum
    max_dd = drawdown.max()

    return {
        "Model": label,
        "CAGR (%)": round(cagr * 100, 2) if pd.notna(cagr) else np.nan,
        "Sharpe": round(sharpe, 2) if pd.notna(sharpe) else np.nan,
        "Sortino": round(sortino, 2) if pd.notna(sortino) else np.nan,
        "Max Drawdown (%)": round(max_dd * 100, 2) if pd.notna(max_dd) else np.nan,
        "Final Equity": round(ending, 2) if pd.notna(ending) else np.nan,
    }


def compute_rolling_sharpe(df: pd.DataFrame, window_days: int = 126) -> pd.Series:
    """Compute rolling Sharpe on daily returns (window in trading days)."""
    r = df["Return"].fillna(0)
    rolling_mean = r.rolling(window=window_days, min_periods=10).mean()
    rolling_std = r.rolling(window=window_days, min_periods=10).std().replace(0, np.nan)
    rolling_sharpe = np.sqrt(252) * rolling_mean / rolling_std
    return rolling_sharpe


# -------------------------
# Core function: build dashboard
# -------------------------
def compare_strategies(
    hybrid_path: str = "results_BTC-USD_phase2_step1.csv",
    ml_path: str = "phase3_ML_signals_BTC.csv",
    ticker: str = "BTC-USD",
    rolling_window: int = 126
) -> pd.DataFrame:
    """
    Load hybrid and ML outputs, compute metrics and visualizations.
    Returns a DataFrame of metrics.
    """
    print("📊 Building ML vs Hybrid Comparison Dashboard...")

    # Load files
    try:
        hybrid = pd.read_csv(hybrid_path, parse_dates=["Date"], index_col="Date")
    except Exception as e:
        raise FileNotFoundError(f"Could not open hybrid file at {hybrid_path}: {e}")

    try:
        ml = pd.read_csv(ml_path, parse_dates=["Date"], index_col="Date")
    except Exception as e:
        # If ML file not present, build a placeholder using Target/Predicted_Label if available in hybrid
        raise FileNotFoundError(f"Could not open ML file at {ml_path}: {e}")

    # Ensure we have necessary columns
    for df in (hybrid, ml):
        if "Close" not in df.columns and "Price" in df.columns:
            df["Close"] = df["Price"]
        if "Return" not in df.columns:
            df["Return"] = df["Close"].pct_change().fillna(0)

    # Prepare hybrid equity curve if missing
    if "Equity_Curve" not in hybrid.columns:
        if "Strategy_Return" in hybrid.columns:
            hybrid["Equity_Curve"] = (1 + hybrid["Strategy_Return"].fillna(0)).cumprod()
        else:
            hybrid["Equity_Curve"] = (1 + hybrid["Return"].fillna(0)).cumprod()

    # Prepare ML equity curve
    if "Equity_Curve" not in ml.columns:
        # Accept common ML columns: Predicted_Label, Pred_Label, Signal
        if "Predicted_Label" in ml.columns:
            ml["Signal"] = ml["Predicted_Label"]
        elif "Pred_Label" in ml.columns:
            ml["Signal"] = ml["Pred_Label"]
        elif "Signal" in ml.columns:
            ml["Signal"] = ml["Signal"]
        elif "Target" in ml.columns:
            ml["Signal"] = ml["Target"]
        else:
            raise ValueError("ML file missing a signal column (Predicted_Label/Signal/Target).")

        ml["Return"] = ml["Close"].pct_change().fillna(0)
        ml["Strategy_Return"] = ml["Signal"].shift(1).fillna(0) * ml["Return"]
        ml["Equity_Curve"] = (1 + ml["Strategy_Return"].fillna(0)).cumprod()

    # Align date ranges (inner join)
    start = max(hybrid.index.min(), ml.index.min())
    end = min(hybrid.index.max(), ml.index.max())
    hybrid = hybrid.loc[start:end].copy()
    ml = ml.loc[start:end].copy()

    # Compute metrics
    hybrid_metrics = calculate_metrics(hybrid, "Hybrid Strategy")
    ml_metrics = calculate_metrics(ml, "ML Strategy")
    metrics_df = pd.DataFrame([hybrid_metrics, ml_metrics]).set_index("Model")

    # Rolling Sharpe
    hybrid_roll_sharpe = compute_rolling_sharpe(hybrid, window_days=rolling_window)
    ml_roll_sharpe = compute_rolling_sharpe(ml, window_days=rolling_window)

    # Correlation / overlap analysis
    # Build a small table of signals (if present) to compare overlap
    def pick_signal_column(d):
        for cand in ("Signal", "Predicted_Label", "Pred_Label", "Target", "Position"):
            if cand in d.columns:
                return cand
        return None

    sig_h = pick_signal_column(hybrid)
    sig_m = pick_signal_column(ml)

    overlap = None
    corr_val = None
    if sig_h and sig_m:
        signals_df = pd.DataFrame({
            "hybrid_signal": hybrid[sig_h].fillna(0).astype(float).reindex(hybrid.index).fillna(0),
            "ml_signal": ml[sig_m].fillna(0).astype(float).reindex(hybrid.index).fillna(0),
        })
        corr_val = signals_df.corr().iloc[0, 1]
        # overlap fraction (when both long)
        both_long = ((signals_df["hybrid_signal"] == 1) & (signals_df["ml_signal"] == 1)).sum()
        overlap_fraction = both_long / len(signals_df) if len(signals_df) else 0.0
        overlap = {"corr": round(corr_val, 3), "overlap_fraction": round(overlap_fraction, 4)}
    else:
        # fallback: compare equity returns correlation
        corr_val = hybrid["Return"].corr(ml["Return"])
        overlap = {"corr": round(corr_val, 3), "overlap_fraction": None}

    # --------------------------
    # Build Plotly Dashboard
    # --------------------------
    fig = make_subplots(
        rows=4, cols=1,
        vertical_spacing=0.08,
        row_heights=[0.45, 0.18, 0.18, 0.19],
        subplot_titles=[
            f"{ticker} — Equity Curve Comparison",
            "Performance Metrics",
            f"Rolling Sharpe (window={rolling_window}d)",
            "Signal Corr / Overlap"
        ]
    )

    # Equity curves
    fig.add_trace(go.Scatter(
        x=hybrid.index, y=hybrid["Equity_Curve"], name="Hybrid Equity", line=dict(color="cyan", width=2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ml.index, y=ml["Equity_Curve"], name="ML Equity", line=dict(color="orange", width=2)
    ), row=1, col=1)

    # Metrics bar chart (two grouped bars)
    metric_names = ["CAGR (%)", "Sharpe", "Sortino", "Max Drawdown (%)", "Final Equity"]
    x = np.arange(len(metric_names))
    fig.add_trace(go.Bar(x=x - 0.18, y=[hybrid_metrics[m] for m in metric_names],
                         name="Hybrid", marker_color="cyan"), row=2, col=1)
    fig.add_trace(go.Bar(x=x + 0.18, y=[ml_metrics[m] for m in metric_names],
                         name="ML", marker_color="orange"), row=2, col=1)
    fig.update_xaxes(tickvals=x, ticktext=metric_names, row=2, col=1)

    # Rolling Sharpe
    fig.add_trace(go.Scatter(x=hybrid_roll_sharpe.index, y=hybrid_roll_sharpe, name="Hybrid Rolling Sharpe",
                             line=dict(color="lightcyan")), row=3, col=1)
    fig.add_trace(go.Scatter(x=ml_roll_sharpe.index, y=ml_roll_sharpe, name="ML Rolling Sharpe",
                             line=dict(color="darkorange")), row=3, col=1)
    # add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    # Signal correlation / overlap info (text + mini heatmap)
    # Show a small textual summary and a heatmap of correlation between Returns & Equity if desired
    info_text = f"Signal correlation: {overlap['corr']}"
    if overlap["overlap_fraction"] is not None:
        info_text += f" · Overlap(long,long): {overlap['overlap_fraction']:.2%}"

    # Add info text as annotation
    fig.add_annotation(dict(
        xref="paper", yref="paper", x=0.01, y=0.06,
        text=f"<b>{info_text}</b>",
        showarrow=False,
        font=dict(size=12)
    ), row=4, col=1)

    # Heatmap between signals/returns
    hm_x = ["Hybrid", "ML"]
    # Use returns and equity correlation
    heat_vals = [
        [1.0, round(hybrid["Return"].corr(ml["Return"]) or 0.0, 3)],
        [round(hybrid["Return"].corr(ml["Return"]) or 0.0, 3), 1.0]
    ]
    fig.add_trace(go.Heatmap(
        z=heat_vals, x=hm_x, y=hm_x, colorscale="Viridis", showscale=True, zmin=-1, zmax=1
    ), row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        title=f"📈 ML vs Hybrid Strategy Comparison — {ticker}",
        height=1100,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.98),
        hovermode="x unified"
    )

    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Metric value", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe", row=3, col=1)
    fig.update_yaxes(title_text="Corr", row=4, col=1)

    fig.show()

    print("\n✅ Metrics summary:")
    print(metrics_df)
    return metrics_df


# -------------------------
# Standalone runner
# -------------------------
if __name__ == "__main__":
    # default file names used by your pipeline
    hybrid_file = "results_BTC-USD_phase2_step1.csv"
    ml_file = "phase3_ML_signals_BTC.csv"

    # Helpful checks
    import os
    if not os.path.exists(hybrid_file):
        raise FileNotFoundError(f"Hybrid file missing: {hybrid_file} — run phase2 step1 first.")
    if not os.path.exists(ml_file):
        raise FileNotFoundError(f"ML signals file missing: {ml_file} — run ML step to create it.")

    compare_strategies(hybrid_file, ml_file)
