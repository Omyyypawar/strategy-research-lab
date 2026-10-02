"""
Phase 2 Visualization Dashboard | Plotly 4-Panel Crypto Chart
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd

pio.renderers.default = "browser"

def plot_dashboard(df, ticker):
    if df is None or df.empty:
        print("⚠️ No data to plot.")
        return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Ensure Bollinger bands exist
    if 'BB_Upper' not in df.columns:
        df['BB_Upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
        df['BB_Lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.15, 0.2, 0.2],
        subplot_titles=[
            f"{ticker} Price (Candlestick + SMA + EMA + Bollinger)",
            "Volume", "RSI (14)", "MACD (12,26,9)"
        ]
    )

    # 1️⃣ Price
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price", increasing_line_color='lime', decreasing_line_color='red'), row=1, col=1)
    for col, color in [('SMA_20','orange'), ('EMA_20','cyan')]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col, line=dict(color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='BB Lower', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

    # Breakouts
    up = df[df['Close'] > df['BB_Upper']]
    down = df[df['Close'] < df['BB_Lower']]
    fig.add_trace(go.Scatter(x=up.index, y=up['Close'], mode='markers', name='Bull Breakout', marker=dict(color='lime', symbol='triangle-up', size=9)), row=1, col=1)
    fig.add_trace(go.Scatter(x=down.index, y=down['Close'], mode='markers', name='Bear Breakout', marker=dict(color='red', symbol='triangle-down', size=9)), row=1, col=1)

    # 2️⃣ Volume
    if 'Volume' in df.columns:
        colors = ['lime' if c >= o else 'red' for c,o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, opacity=0.6, name='Volume'), row=2, col=1)

    # 3️⃣ RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='violet', width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # 4️⃣ MACD
    if {'MACD_Line','MACD_Signal','MACD_Hist'}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Line'], mode='lines', name='MACD', line=dict(color='cyan')), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal', line=dict(color='orange')), row=4, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', marker_color='gray', opacity=0.4), row=4, col=1)

    fig.update_layout(
        template='plotly_dark', title=f"{ticker} Technical Dashboard",
        height=1150, xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    fig.show()
