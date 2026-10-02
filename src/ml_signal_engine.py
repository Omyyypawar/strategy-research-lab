"""
Phase 3 · Step 3 — Advanced ML Signal Engine with Visualization
Author: Om | Strategy Research Lab
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import plotly.io as pio

pio.renderers.default = "browser"


def train_ml_model(df: pd.DataFrame):
    """Train ML model and produce analysis visuals."""
    print("🤖 Training ML Signal Engine...")

    if df.empty:
        raise ValueError("❌ The input dataframe is empty.")
    if "Close" not in df.columns:
        raise KeyError("❌ Missing 'Close' column in data.")

    # Use existing target if present
    if "Target" not in df.columns:
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    else:
        print("ℹ️ Using existing Target column.")

    # Drop missing rows in essential indicators
    key_cols = ["Close", "RSI", "SMA_20", "EMA_20", "MACD_Line"]
    df = df.dropna(subset=[c for c in key_cols if c in df.columns])

    features = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["Target", "Return", "Strategy_Return", "Equity_Curve"]
    features = [f for f in features if f not in exclude]

    X, y = df[features], df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    df["Predicted_Label"] = model.predict(X)
    preds_test = model.predict(X_test)
    acc = accuracy_score(y_test, preds_test)
    print(f"✅ Model trained. Test Accuracy: {acc*100:.2f}%")
    print("\n", classification_report(y_test, preds_test))

    # ---- 1️⃣ Feature Importance Heatmap ----
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    top_feats = importances.head(15)

    plt.figure(figsize=(8, 5))
    sns.heatmap(top_feats.values.reshape(1, -1), annot=True, fmt=".2f",
                yticklabels=["Importance"], xticklabels=top_feats.index, cmap="YlGnBu")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

    # ---- 2️⃣ Rolling Accuracy Tracker ----
    window = 100
    df["Correct"] = (df["Predicted_Label"] == df["Target"]).astype(int)
    df["Rolling_Accuracy"] = df["Correct"].rolling(window).mean() * 100

    # ---- 3️⃣ Confusion Matrix Heatmap ----
    cm = confusion_matrix(y_test, preds_test)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ---- 4️⃣ Interactive Plotly Dashboard ----
    fig = make_subplots(rows=2, cols=1, subplot_titles=[
        "Rolling Accuracy (Window=100 Days)",
        "Top 15 Feature Importances"
    ])

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Rolling_Accuracy"], mode="lines", name="Rolling Accuracy (%)",
        line=dict(color="lime")
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=top_feats.index, y=top_feats.values, name="Feature Importance", marker_color="cyan"
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        title="📈 ML Signal Engine Performance Dashboard",
        height=800,
        hovermode="x unified"
    )
    fig.show()

    return df, model


if __name__ == "__main__":
    print("🚀 Starting Phase 3 – Step 3: ML Intelligence Dashboard")

    input_file = "phase2_features_BTC.csv"
    output_file = "phase3_ML_signals_BTC.csv"

    try:
        df = pd.read_csv(input_file, parse_dates=["Date"], index_col="Date")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Input file {input_file} not found.")

    df_ml, model = train_ml_model(df)

    df_ml.to_csv(output_file)
    joblib.dump(model, "phase3_rf_model.pkl")

    print(f"\n💾 Saved ML signals to {output_file}")
    print("✅ Phase 3 – Step 3 complete! Dashboard generated successfully.")
